# tools

model_convert
  - torch2onnx.py：pytorch模型转onnx模型，并且保留模型的逐层尺寸信息

onnx_analyzer
  - onnx_analyzer.py：读取onnx模型，并打印每层的type、name、in/out shape、kernel信息
