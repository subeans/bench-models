import onnx
import onnxoptimizer 
from onnx import helper
import onnxruntime as ort
import numpy as np
import argparse
import time 


def make_dataset(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def original_onnx_serving(model_name,batch_size,size,repeat=100):
    model_path = f"./{model_name}_{batch_size}.onnx"
    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
        
    data, image_shape = make_dataset(batch_size,size)
        
    time_list = []
    for i in range(repeat):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        # print(f"ONNX serving {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)

    with open(f"./results/{model_name}_{batch_size}.txt", "w") as output:
        output.write(str(time_list))
    time_mean = np.mean(np.array(time_list))
    time_medium = np.median(np.array(time_list))
    print(f"{model_name} inference time medium : {time_medium*1000} ms")
    print(f"{model_name} inference time mean : {time_mean*1000} ms")

def optimize_onnx(model_name,batch_size,skip_fuse_bn=False):
    model_path = f"./{model_name}_{batch_size}.onnx"
    opt_onnx = f'{model_name}_{batch_size}.opt.onnx'

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    optimizers_list = onnxoptimizer.get_fuse_and_elimination_passes()
    if skip_fuse_bn:
        optimizers_list.remove('fuse_bn_into_conv')
    # print(optimizers_list)
    model = onnxoptimizer.optimize(model, optimizers_list,
                                   fixed_point=True)
    onnx.checker.check_model(model)
    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())


def optimize_onnx_serving(model_name,batch_size,size,repeat=100):
    optimize_onnx(model_name,batch_size)
    model_path = f"./{model_name}_{batch_size}.opt.onnx"
    # print(model_path)
    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
        
    data, image_shape = make_dataset(batch_size,size)
        
    time_list = []
    for i in range(repeat):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        # print(f"ONNX serving {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)

    with open(f"./results/optimized_{model_name}_{batch_size}.txt", "w") as output:
        output.write(str(time_list))
    time_mean = np.mean(np.array(time_list))
    time_medium = np.median(np.array(time_list))
    print(f"{model_name} optimized inference time medium : {time_medium*1000} ms")
    print(f"{model_name} optimized inference time mean : {time_mean*1000} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--repeat',default=100 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    repeat = args.repeat


    img_size=224
    if model_name == "inception_v3":
        img_size = 299 

    original_onnx_serving(model_name,batch_size,img_size,repeat)
    optimize_onnx_serving(model_name,batch_size,img_size,repeat)