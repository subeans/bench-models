import onnx
from onnx import helper
import onnxruntime as ort
import numpy as np
import pandas as pd
import json
import argparse
import time 
from collections import OrderedDict
from onnxruntime import InferenceSession, SessionOptions, get_device
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession

from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    SessionIOBinding, OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)


if get_device().upper() == 'GPU':
    ort_device = C_OrtDevice(
        C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
    provider = 'CUDAExecutionProvider'
else:
    ort_device = C_OrtDevice(
        C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    provider = 'CPUExecutionProvider'

print("provider = %r" % provider)


def make_dataset(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def original_onnx_serving(model_name,batch_size,size,repeat=10):
    model_path = f"./{model_name}_{batch_size}.onnx"
    so = SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(model_path, so, providers=[provider])
    # bind = SessionIOBinding(session._sess)

    print("graph_optimization_level:", so.graph_optimization_level)

    # session = ort.InferenceSession(model_path)
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

    prof = session.end_profiling()
    with open(prof, "r") as f:
        js = json.load(f)
        df = pd.DataFrame(OnnxWholeSession.process_profiling(js))
        print(df)
    time_mean = np.mean(np.array(time_list[1:]))
    time_medium = np.median(np.array(time_list[1:]))
    print(f"{model_name} inference time medium : {time_medium*1000} ms")
    print(f"{model_name} inference time mean : {time_mean*1000} ms")


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
