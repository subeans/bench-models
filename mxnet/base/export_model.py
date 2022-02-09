import json
from pyexpat import model

from charset_normalizer import models

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np


ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

def download_model(model_name,imgsize=224):
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

    input_shape = (1, 3, imgsize, imgsize)
    data = np.random.uniform(size=input_shape)

    input_data = mx.nd.array(data, ctx=ctx)

    print("-"*10,f"{model_name} Parameter Info","-"*10)
    print(model.summary(input_data))

    sym = model(mx.sym.var('data'))
    if isinstance(sym, tuple):
        sym = mx.sym.Group([*sym])

    target_path = f"./{model_name}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    with open(f'./{model_name}/model.json', "w") as fout:
        fout.write(sym.tojson())
    model.collect_params().save(f'./{model_name}/model.params')

    print("-"*10,f"Download {model_name} complete","-"*10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)

    args = parser.parse_args()

    model_name = args.model
    img_size = 224
    if args.model == "all":
        models = ["mobilenet", "mobilenet_v2", "inception_v3","resnet50","alexnet","vgg16","vgg19"]
    else:
        models = [args.model]

    for model in models:
        if model == 'inception_v3':
            img_size = 299
        download_model(model,img_size)
        