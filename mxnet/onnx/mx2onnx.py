import numpy as np
from mxnet.contrib import onnx as onnx_mxnet


def convert(model_name,size):
    input_shape=(1,3,size,size)
    #convert using onnx
    target_path = f"./{model_name}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    onnx_file=f'./{model_name}/{model_name}.onnx'

    params = f'../base/{model_name}/model-0000.params'
    sym=f'../base/{model_name}/model-symbol.json'
    onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
    print('onnx export done')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    args = parser.parse_args()
    model_name = args.model

    img_size=224
    if model_name == "inception_v3":
        img_size = 299 
    convert(model_name,img_size)
