# Steps to train EfficientDet
# Convert EfficientDet to TensorRT

###  1. TRAINING EFFICIENT-DET using zylo117's repository
# clone the git repo by zylo117
1. git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
2. cd Yet.... #into the folder
3. clone this repo inside
4. create virtual env and install all the requirements
5. Use effdet_train.ipynb to train (can be used on colab/kaggle)
6. Model is saved after number of specified steps and for keyboard interrupt

###  2. Convert pth to onnx.
1. git clone 'https://github.com/murdockhou/Yet-Another-EfficientDet-Pytorch-Convert-ONNX-TVM.git'
2. Use convert/convert_onnx.py to convert to onnx.

### 3 Convert to tensorRT
The above conversion is still complex for tensorRT therefore we need to simplify it.
1. git clone https://github.com/daquexian/onnx-simplifier.git
2. cd into dir
### use the following code
      import onnx
      from onnxsim import simplify

      onnx_path = r'./efficientdet-d1-0610.onnx'
      output_path = r'./effdet_simp_0610.onnx'

      model = onnx.load(onnx_path)
      model_simp, check = simplify(model)
      onnx.save(model_simp,output_path)

This simplifies many operations which are otherwise impossible to convert to tensorRT directly.
3. Run convert_onnx2engine.py to convert to .engine file


### INSTRUCTIONS TO RUN trt_inference.py
1. copy trt_inference.py to Yet_another... DIR as some functions are used from there
1. Run trt_inference.py to check model performance
Currently the script looks for all png files inside new_test folder (need to create does not exist) and runs inference and displays image when it detects class.


########### TESTED ON #########
nvidia-tensorrt==8.0.0.3
torchvision==0.9.1
torch==1.8.1
