# mxnet-serving
MXNET Inference Serving ( baseline ) 

## Workloads
### image classification 
- Running from source 
```
pip3 install mxnet 
git clone https://github.com/subeans/tvm-model.git
cd tvm-model/mxnet/base 

# 1. download model from gluon zoo and export model ( /model_name/model.params and model.json )
# models -> mobilenet , mobilenet_v2 , inception_v3, resnet50 , alexnet , vgg16 , vgg19
python3 export_model.py --model resnet50 

# 2. load model and inference 
python3 load_serving.py --model resnet50

# 3. upload model to s3 
aws s3 sync ./resnet50/ s3://BUCKET_PATH/FOLDER/
```
### NLP 
- Running from source 
```
# 1. download model from gluon zoo and export model ( /model_name/model.params and model.json )
# models -> bert_base , distilbert 
# python3 bert_export_model.py --model bert_base 

# 2. load model and inference 
# python3 bert_load_serving.py --model bert_base

# 3. upload model to s3 
aws s3 sync ./bert_base/ s3://BUCKET_PATH/FOLDER/
