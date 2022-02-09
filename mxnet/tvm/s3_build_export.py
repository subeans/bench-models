import os
import argparse
import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_executor as runtime
from tvm.contrib import graph_runtime
import mxnet as mx
import warnings 
from mxnet import gluon

import tvm
from tvm import relay

mx_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

def load_model(model_name):
    model_json = f"./{model_name}/model.json"
    model_params = f"./{model_name}/model.params"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gluon.nn.SymbolBlock.imports(model_json, ['data'], model_params, ctx=mx_ctx)
    return model 

def get_tvm_format(model_name, batch_size, imgsize = 224, dtype="float32", layout="NCHW"):
    input_name = "data"
    input_shape = (batch_size, 3, imgsize, imgsize)
    output_shape = (batch_size, 1000)

    block = load_model(model_name)

    mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
    if layout == "NHWC":
        mod = convert_to_nhwc(mod)
    else:
        assert layout == "NCHW"
   
    return mod, params, input_name, input_shape, output_shape


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod


def export_results(graph,lib,params,model_name,batch_size):
    target_path = f"./{model_name}_{batch_size}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    name = f"./{model_name}_{batch_size}/model"
    graph_fn, lib_fn, params_fn = [name+ext for ext in ('.json','.tar','.params')]
    lib.export_library(lib_fn)
    with open(graph_fn, 'w') as f:
        f.write(graph)
    with open(params_fn, 'wb') as f:
        f.write(relay.save_param_dict(params))


def benchmark(model_name, batch_size, target,dtype='float32', repeat=3):
    layout = "NCHW"
    mod, params, input_name, input_shape, output_shape = get_tvm_format(
        model_name, batch_size, dtype, layout
    )
    ctx = tvm.cpu()

    # export only tar format 
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    # lib.export_library(f"./{model}_{batch_size}.tar")
    # module = runtime.GraphModule(lib["default"](ctx))

    # export graph , lib , params   
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target=target, params=params)

    export_results(graph,lib,params,model_name,batch_size)

    module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    module.load_params(params)     

    # Feed input data
    data = np.random.uniform(size=input_shape)
    module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--target',default='llvm -mcpu=core-avx2' , type=str)

    args = parser.parse_args()

    model_name = args.model
    batchsize = args.batchsize
    target = args.target

    benchmark(model_name,batchsize,target)
