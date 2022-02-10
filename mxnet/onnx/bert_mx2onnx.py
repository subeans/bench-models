import mxnet as mx 
import numpy as np
import os 


def convert(model_name,batch_size,seq_length,dtype="float32"):
    sym = f'./{model_name}/model-symbol.json'
    params =  f'./{model_name}/model-0000.params'

    onnx_file = f"./{model_name}.onnx"
    target_path = f"./{model_name}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    if model_name=="bert_base":
        input_shape=[(batch_size,seq_length),(batch_size,seq_length),(batch_size,)]
    elif model_name == "distilbert":
        input_shape=[(batch_size,seq_length),(batch_size,)]

    converted_model_path = mx.onnx.export_model(sym, params, input_shape, np.float32 , onnx_file)
    print("mxnet model from s3 to onnx : done")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert_base' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    seq_length = args.seq_length

    convert(model_name,batch_size,seq_length)
