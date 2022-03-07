import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
import tvm
from tvm import relay
# import tvm.contrib.graph_runtime as runtime
import tvm.contrib.graph_executor as runtime



import tvm.testing
import warnings
from mxnet import gluon

warnings.filterwarnings(action='ignore') 

def load_model(model_name):
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

    model_json = f"../base/{model_name}_{batch_size}/model-symbol.json"
    model_params = f"../base/{model_name}_{batch_size}/model-0000.params"


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1'], model_params, ctx=ctx)
    return model 

def compile_tvm(model_name,batch_size,seq_length):

    # load origianl mxnet model 
    model = load_model(model_name)

    # Prepare input data
    dtype = "float32"
    inputs = np.random.randint(0, 2000, size=(seq_length,batch_size)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

    shape_dict = {
            'data0': (seq_length,batch_size),
            'data1': (batch_size,)
        }

    mod, params = relay.frontend.from_mxnet(model, shape_dict)

    # Compile the imported model
    # target = "llvm -mcpu=core-avx2"
    target = tvm.target.arm_cpu()
    # with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    #     graph, lib, cparams = relay.build(mod, target, params=params)

    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(f"./{model_name}_{batch_size}.tar")

    dev = tvm.cpu()
    module = runtime.GraphModule(lib["default"](dev))
    
    module.set_input(data0=inputs, data1=valid_length)

    # Evaluate
    ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=5)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM {model_name} latency for batch {batch_size} : {np.mean(prof_res):.2f} ms")

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

    compile_tvm(model_name,batch_size,seq_length)
