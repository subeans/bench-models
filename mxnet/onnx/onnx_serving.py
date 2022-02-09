import numpy as np
import onnxruntime as ort

import argparse
import time
import warnings
warnings.filterwarnings(action='ignore')

def make_dataset(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape


def onnx_serving(model_name,batch_size,size):
    model_path = f"./{model_name}/{model_name}.onnx"
        
    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
        
    data, image_shape = make_dataset(batch_size,size)
        
    time_list = []
    for i in range(5):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        print(f"ONNX serving {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)

    time_medium = np.median(np.array(time_list))
    print('{} median latency (batch={}): {} ms'.format(model_name, batch_size, time_medium * 1000))
    print('{} mean latency (batch={}): {} ms'.format(model_name, batch_size, np.mean(np.array(time_list)*1000)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize 

    img_size = 224
    if args.model == "all":
        models = ["mobilenet", "mobilenet_v2", "inception_v3","resnet50","alexnet","vgg16","vgg19"]
    else:
        models = [args.model]

    for model in models:
        if model == 'inception_v3':
            img_size = 299
        onnx_serving(model,batch_size,img_size)
