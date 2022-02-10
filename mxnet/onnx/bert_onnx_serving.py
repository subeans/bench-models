import numpy as np
import onnxruntime as ort
import argparse
import time
import warnings

warnings.filterwarnings(action='ignore')

def benchmark(model_name,batch_size,seq_length,dtype="float32",c=5):
    # Prepare input data
    inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

    # Convert to MXNet NDArray and run the MXNet model

    model_path = f"./{model_name}.onnx"

    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]

    if model_name=="bert_base":
        input_info = {inname[0]: inputs,inname[1]:token_types,inname[2]:valid_length}
    elif model_name == "distilbert":
        input_info = {inname[0]: inputs,inname[1]:valid_length}
        
    time_list = []
    for i in range(batch_size):
        start_time = time.time()
        session.run(outname,input_info )
        running_time = time.time() - start_time
        print(f"ONNX serving {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)

    time_medium = np.median(np.array(time_list))
    print('{} median latency (batch={}): {} ms'.format(model_name, batch_size, time_medium * 1000))
    print('{} mean latency (batch={}): {} ms'.format(model_name, batch_size, np.mean(np.array(time_list)*1000)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batchsize = args.batchsize
    seq_length=args.seq_length

    benchmark(model_name,batchsize,seq_length)
