import os 
import argparse
import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_executor as runtime
from tvm.contrib import graph_runtime
import mxnet as mx

import tvm
from tvm import relay

from mxnet.gluon.model_zoo import vision
import mxnet 

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

def get_network(model_name, imgsize,batch_size, target,dtype="float32", layout="NCHW"):
    """Get the symbol definition and random weight of a network"""
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    models_detail = {
        'mobilenet':vision.mobilenet0_5(pretrained=True, ctx=ctx),
        'mobilenet_v2':vision.get_mobilenet_v2(1, pretrained=True),
        'inception_v3':vision.inception_v3(pretrained=True, ctx=ctx),
        'resnet50': vision.get_resnet(1, 50, pretrained=True),
        'alexnet':vision.alexnet(pretrained=True,ctx=ctx),
        'vgg16':vision.vgg16(pretrained=True, ctx=ctx),
        'vgg19':vision.vgg19(pretrained=True, ctx=ctx)
    }

    model = models_detail[model_name]
    model.hybridize()

    # hybridize 후 한번은 foward path 진행하야함 
    input_shape = (1, 3, imgsize, imgsize)
    data = np.random.uniform(size=input_shape)

    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    mx_data = mx.nd.array(data_array)
    model(mx_data)

    # mxnet to tvm format
    mod, params = relay.frontend.from_mxnet(
            model, shape={"data": input_shape}, dtype=dtype
        )

    if layout == "NHWC":
        mod = convert_to_nhwc(mod)
    else:
        assert layout == "NCHW"

    # tvm compile 
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(f"./{model_name}_{batch_size}.tar")
    
    print("export done :",f"./{model_name}_{batch_size}.tar")
    #########################################################################################
    # TEST INFERENCE 
    #########################################################################################
    # Make module
    # dev = tvm.cpu()
    # module = runtime.GraphModule(lib["default"](dev))

    # data = np.random.uniform(size=input_shape)
    # module.set_input(input_name, data)

    # # Evaluate
    # ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=5)
    # prof_res = np.array(ftimer().results) * 1000
    # print(f"TVM {model_name} latency for batch {batch_size} : {np.mean(prof_res):.2f} ms")
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--target',default='llvm -mcpu=core-avx2' , type=str)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    target = args.target 
    img_size = 224
    if args.model == "all":
        models = ["mobilenet", "mobilenet_v2", "inception_v3","resnet50","alexnet","vgg16","vgg19"]
    else:
        models = [args.model]

    for model in models:
        if model == 'inception_v3':
            img_size = 299
        get_network(model, img_size,batch_size, target)
