import numpy as np
import tensorrt as trt
import argparse
import time
import os
import torch
import cv2
import sys
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes, Anchors
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

def display_img(img_path,rois,scores,class_ids):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(640,640))

    for i in range(len(rois)):
        temp = rois[i]
        xmin = int(temp[0])
        ymin = int(temp[1])
        xmax = int(temp[2])
        ymax = int(temp[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        str_label = obj_list[class_ids[i]]
        
        cv2.putText(img, str_label, (xmin, ymin-2),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return img


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
        
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def get_engine(engine_file_path):
    if os.path.isfile(engine_file_path):
        with open(engine_file_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )
            if engine == None:
                exit()
        return engine

def load_image(img, pagelocked_buffer):
    # Select an image at random to be the test case.
    np.copyto(pagelocked_buffer, img)

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.

    return [out.host for out in outputs]


def run(framed_imgs, engine, inputs, outputs, bindings, stream, context, batch_size):
    load_image(framed_imgs, pagelocked_buffer=inputs[0].host)
    regression, classification = do_inference(context, bindings=bindings, inputs=inputs,outputs=outputs,stream=stream, batch_size=batch_size)
    # x_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(1))
    # x = x.reshape(x_shape)

    regression_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(1))
    regression = regression.reshape(regression_shape)
    classification_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(2))
    classification = classification.reshape(classification_shape)
    # return x, regression, classification
    return  regression, classification

input_size = 640
logger = trt.Logger(trt.Logger.INFO)
compound_coef = 1
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.4
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
color_list = standard_to_bgr(STANDARD_COLORS)
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reverse_dict ={1: 'stop',2: 'speedLimitUrdbl',3: 'speedLimit25',4: 'pedestrianCrossing',5: 'speedLimit35',6: 'turnLeft',
 7: 'slow', 8: 'speedLimit15',9: 'speedLimit45',10: 'rightLaneMustTurn', 11: 'signalAhead', 12: 'keepRight', 13: 'laneEnds', 14: 'school', 15: 'merge', 16: 'addedLane', 
 17: 'rampSpeedAdvisory40', 18: 'rampSpeedAdvisory45', 19: 'curveRight',20: 'speedLimit65', 21: 'truckSpeedLimit55', 22: 'thruMergeLeft', 23: 'speedLimit30',24: 'stopAhead',25: 'yield',26: 'thruMergeRight', 27: 'dip',28: 'schoolSpeedLimit25', 29: 'thruTrafficMergeLeft',
 30: 'noRightTurn', 31: 'rampSpeedAdvisory35', 32: 'curveLeft',33: 'rampSpeedAdvisory20', 34: 'noLeftTurn',35: 'zoneAhead25', 36: 'zoneAhead45', 37: 'doNotEnter',
 38: 'yieldAhead', 39: 'roundabout', 40: 'turnRight',41: 'speedLimit50', 42: 'rampSpeedAdvisoryUrdbl', 43: 'rampSpeedAdvisory50', 44: 'speedLimit40', 45: 'speedLimit55', 46: 'doNotPass',47: 'intersection'}


engine_path = r'./weights/efficientdet0610.engine'
DIR = r'./new_test/'
batch_size = 1
num_classes = 48

img_paths = [DIR+i for i in os.listdir(DIR) if i.split('.')[-1]=='png']

for j in img_paths:
    framed_imgs = cv2.imread(j).astype(np.float32)
    framed_imgs = cv2.cvtColor(framed_imgs,cv2.COLOR_BGR2RGB)
    framed_imgs = cv2.resize(framed_imgs,(640,640))
    normalized_img = [(framed_imgs[..., ::-1] / 255 - mean) / std]
    
    torch_img = torch.as_tensor(normalized_img)
    torch_img = torch.as_tensor(torch_img).permute(0, 3, 1, 2)
    final_reshape = torch_img.numpy().reshape(-1)

    # print(np.shape(reshape_img))


    engine = get_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    t1 = int(round(time.time() * 1000))
    regression, classification = run(final_reshape, engine, inputs, outputs, bindings, stream, context, batch_size)

    t2 = int(round(time.time() * 1000))
    print('modeltime: ', t2-t1, 'ms')

    anchors = Anchors(anchor_scale=4.0,pyramid_levels=(torch.arange(5) + 3).tolist(),compound_coef=compound_coef, num_classes=num_classes,ratios=anchor_ratios, scales=anchor_scales)
    anchors = anchors(torch_img, torch.float32)

    regression = torch.from_numpy(regression[0])
    classification = torch.from_numpy(classification[0])
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(torch_img,anchors, regression, classification,regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    # out is list of 1 dictionery
    pred_dict = out[0]
    rois = pred_dict['rois']
    class_ids = pred_dict['class_ids']
    scores = pred_dict['scores']
    if len(rois)>0:
        nms_indices = torchvision.ops.nms(torch.as_tensor(rois),torch.as_tensor(scores),iou_threshold=iou_threshold)
        if len(nms_indices)==1:
            rois = [rois[nms_indices]]
            scores = [scores[nms_indices]]
            class_ids = [class_ids[nms_indices]]
            #  print(class_ids)
        else:
            rois = rois[nms_indices]
            scores = scores[nms_indices]
            class_ids = class_ids[nms_indices]
        print(rois)
        print(scores)
        print(class_ids)
        print(j)
        fimg = display_img(j,rois,scores,class_ids)
        plt.imshow(fimg)
        plt.show()
        cv2.waitKey(0)
        print(j)
