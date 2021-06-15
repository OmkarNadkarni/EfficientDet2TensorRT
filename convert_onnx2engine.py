import tensorrt
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

weights_path = r'./effdet_d1_simple.onnx'
engine_file_path = r'./efficientdet_d1.engine'

# onnx_model = onnx.load(weights_path)
# onnx.checker.check_model(onnx_model)



# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder_config = builder.create_builder_config()  

    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
        #print(model.read())

    print('Completed parsing of ONNX file')

    last_layer = network.get_layer(network.num_layers - 1)
# Check if last layer recognizes it's output
    if not last_layer.get_output(0):
    # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))

# generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    print(network)
    engine = builder.build_engine(network,builder_config)

    #context = engine.create_execution_context()
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

    return engine, context


engine, context = build_engine(weights_path)
