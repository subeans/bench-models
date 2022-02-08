import numpy as np
import time 
import argparse
import tvm.contrib.graph_executor as runtime

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te

def benchmark(model, batch_size, repeat):
    ctx = tvm.cpu()

    if model in ["bert"]:

        loaded_lib = tvm.runtime.load_module(f'/home/ec2-user/{framework}/{model}.tar')
        module = runtime.GraphModule(loaded_lib["default"](ctx))

        # Feed input data
        dtype = "float32"
        seq_length = 128
        data = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

        # Feed input data version2 
        # shape_dict = {
        #     "data0": (batch_size, seq_length),
        #     "data1": (batch_size, seq_length),
        #     "data2": (batch_size,),
        # }
        # input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])
        # seq_length = input_shape[0][1]
        # data = np.random.uniform(size=input_shape[0])
        # token_types = np.random.uniform(size=input_shape[1])
        # valid_length = np.array([seq_length] * batch_size)


        module.set_input(data0=data, data1=token_types, data2=valid_length)

    else:
        input_name = "data"
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 1000)
        
        loaded_lib = tvm.runtime.load_module(f'/home/ec2-user/{framework}/{model}_{batch_size}.tar')

        module = runtime.GraphModule(loaded_lib["default"](ctx))

        # Feed input data
        data = np.random.uniform(size=input_shape)
        module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)

def make_network_key(network_name, batch_size):
    return "model:%s-batchsize:%s" % (network_name, batch_size)
    
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--framework', type=str, default='mxnet')
    parser.add_argument("--repeat", type=int, default=3)

    args = parser.parse_args()
    model = args.model
    framework = args.framework


    if args.model == "all":
        models = ["resnet50", "mobilenet_v2", "bert"]
    else:
        models = [args.model]
    batch_sizes = [args.batchsize]
    
    # Benchmark
    result_messages = []
    for model in models:
        for batch_size in batch_sizes:
            network_key = make_network_key(model, batch_size)
            print("Benchmark %s ..." % network_key)

            prof_res = benchmark(model, batch_size, args.repeat)

            prof_res *= 1000  # convert to millisecond
            message = "%-18s %-12s %-19s (%s)" % (
                    model,
                    batch_size,
                    "%.2f ms" % np.mean(prof_res),
                    "%.2f ms" % np.std(prof_res),
                )
            result_messages.append(message)

    # Print result
    print("-------------------------------------------------------------")
    print(
        "%-18s %-12s %-20s"
        % ("Model Name", "Batch size", "Mean Inference Time (std dev)")
    )
    print("-------------------------------------------------------------")
    for line in result_messages:
        print(line)
    print("-------------------------------------------------------------")
