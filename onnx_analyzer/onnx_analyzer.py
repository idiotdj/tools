#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import onnx
import onnx.shape_inference
import itertools
from tabulate import tabulate

def print_model_ops_and_params(onnx_model_path):
    # 加载ONNX模型
    model = onnx.load(onnx_model_path)
    # 使用onnx.shape_inference推理shape信息
    model_with_shapes = onnx.shape_inference.infer_shapes(model)
    # 重新保存包含shape信息的ONNX模型
    # onnx.save(model_with_shapes, os.path.splitext(onnx_model_path)[0] + "_with_shapes.onnx")
    graph = model_with_shapes.graph

    # 创建一个字典来存储所有变量的维度信息
    shape_dict = {}
    for value_info in itertools.chain(graph.input, graph.output, graph.value_info):
        shape = [d.dim_value if d.dim_value != 0 else '?' for d in value_info.type.tensor_type.shape.dim]
        shape_dict[value_info.name] = shape

    # 表格数据
    table_data = []
    
    # 遍历模型的每个节点并打印操作（OP）和操作的参数
    for node in graph.node:
        # 获取节点的输入和输出名称
        input_names = node.input
        output_names = node.output

        # 获取输入和输出的尺寸
        input_shapes = [shape_dict.get(name, []) for name in input_names]
        output_shapes = [shape_dict.get(name, []) for name in output_names]

        # 处理多余的`[]`
        input_shapes_str = ', '.join(f'[{", ".join(map(str, shape))}]' for shape in input_shapes if shape)
        output_shapes_str = ', '.join(f'[{", ".join(map(str, shape))}]' for shape in output_shapes if shape)

        # 如果是卷积层，打印额外的属性
        if node.op_type == "Conv":
            kernel_shape, strides, pads = None, None, None
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                if attr.name == "strides":
                    strides = attr.ints
                if attr.name == "pads":
                    pads = attr.ints
            table_data.append([node.op_type, node.name, input_shapes_str, output_shapes_str, f"k={kernel_shape}, s={strides}, p={pads}"])
        elif node.op_type != "QIRQuantize":
            table_data.append([node.op_type, node.name, input_shapes_str, output_shapes_str, ""])

    # 使用tabulate库输出表格
    headers = ["OP", "Name", "Input Shape", "Output Shape", "Attr"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
if __name__ == "__main__":
    # 通过命令行参数传递模型路径
    if len(sys.argv) != 2:
        print("请输入正确的参数: python script.py [模型路径]")
        sys.exit(1)
    
    onnx_model_path = sys.argv[1]

    # onnx_model_path = "./resnet50.onnx"
    
    # 调用函数打印模型的操作和参数
    print_model_ops_and_params(onnx_model_path)