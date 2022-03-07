import mxnet as mx 
import numpy as np
import onnx
import os 


def convert(model_name,batch_size,seq_length,dtype="float32"):
    sym = f'../base/{model_name}_{batch_size}/model-symbol.json'
    params =  f'../base/{model_name}_{batch_size}/model-0000.params'

    onnx_file = f"./{model_name}.onnx"
    target_path = f"./{model_name}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    input_shape=[(seq_length,batch_size),(batch_size,)]
    
    converted_model_path = mx.onnx.export_model(sym, params, input_shape, np.float32 , onnx_file)
    print("mxnet model from s3 to onnx : done")
    return converted_model_path

def change_version(model_path):
    model = onnx.load(model_path)

    op = onnx.OperatorSetIdProto()
    op.version = 13
    update_model = onnx.helper.make_model(model.graph, opset_imports=[op])
    onnx.save(update_model, model_path)
    print("convert onnx version and save")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='lstm' , type=str)
    parser.add_argument('--batchsize',default=8 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    seq_length = args.seq_length

    path = convert(model_name,batch_size,seq_length)
    change_version(path)
