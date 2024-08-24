#!/usr/bin/python3

import torch
import torchvision.models as models
import onnx
import onnx.shape_inference

# 定义模型名称
model_name = "resnet18"

# 加载预训练的模型
model = getattr(models, model_name)(pretrained=True)
model.eval()

# 设置输入维度，例如：batch_size=1, channels=3, height=224, width=224
x = torch.randn(1, 3, 224, 224)

# 导出模型为ONNX
onnx_file = f"{model_name}.onnx"
onnx_file_with_shapes = f"{model_name}_with_shapes.onnx"
torch.onnx.export(model,                    # model being run
                  x,                        # model input (or a tuple for multiple inputs)
                  onnx_file,                # where to save the model (can be a file or file-like object)
                  export_params=True,       # store the trained parameter weights inside the model file
                  opset_version=11,         # the ONNX version to export the model to
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names=['input'],    # the model's input names
                  output_names=['output'])  # the model's output names

# 使用onnx.shape_inference推理shape信息并保存模型
onnx_model = onnx.load(onnx_file)
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

# 保存包含shape信息的ONNX模型
onnx.save(onnx_model, onnx_file_with_shapes)

# 检查并打印每一层的输入输出shape
graph = onnx_model.graph
for node in graph.node:
    print(f"Node: {node.name}")
    for input_tensor in node.input:
        # 查找输入tensor的shape
        input_shape = None
        for value_info in graph.value_info:
            if value_info.name == input_tensor:
                input_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                break
        if input_shape is None:
            for input_info in graph.input:
                if input_info.name == input_tensor:
                    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                    break
        print(f"Input: {input_tensor}, Shape: {input_shape}")
    for output_tensor in node.output:
        # 查找输出tensor的shape
        output_shape = None
        for value_info in graph.value_info:
            if value_info.name == output_tensor:
                output_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                break
        if output_shape is None:
            for output_info in graph.output:
                if output_info.name == output_tensor:
                    output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
                    break
        print(f"Output: {output_tensor}, Shape: {output_shape}")
