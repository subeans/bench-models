import numpy as np
from mxnet.contrib import onnx as onnx_mxnet


def convert(model_name):
    input_shape=(1,3,224,224)
    #convert using onnx
    onnx_file=f'./{model_name}/{model_name}.onnx'
    params = f'./{model_name}/model-0000.params'
    sym=f'./{model_name}/model-symbol.json'
    onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
    print('onnx export done')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    args = parser.parse_args()
    model_name = args.model


    convert(model_name)
