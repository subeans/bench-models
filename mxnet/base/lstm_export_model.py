import warnings
warnings.filterwarnings('ignore')

import random
import time
import multiprocessing as mp
import numpy as np
import gluonnlp as nlp

import mxnet as mx
from mxnet import nd, gluon, autograd

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""
    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, data, valid_length):
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                    F.expand_dims(valid_length, axis=1))
        return agg_state

class SentimentNet(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None # will set with lm embedding later
            self.encoder = None # will set with lm encoder later
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(1, flatten=False))

    def hybrid_forward(self, F, data, valid_length): 
        embedded = self.embedding(data)
        encoded = self.encoder(embedded)
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out

def get_model(model_name,batchsize,seq_length,dtype="float32"):
    model, vocab = nlp.model.get_model(name='standard_lstm_lm_200',
                                        dataset_name='wikitext-2',
                                        pretrained=True,
                                        ctx=mx.cpu(),
                                        dropout=0)
    net = SentimentNet()

    # Use Pretrained Embeddings from wikitext-2
    net.embedding = model.embedding
    # Use Pretrained Encoder states (LSTM) from wikitext-2
    net.encoder = model.encoder

    net.output.initialize(ctx=mx.cpu())
    net.hybridize()

    inputs = np.random.randint(0, 2000, size=(seq_length,batchsize)).astype(dtype)
    valid_length = np.asarray([seq_length] * batchsize).astype(dtype)
            
    inputs_nd = mx.nd.array(inputs, ctx=ctx)
    valid_length_nd = mx.nd.array(valid_length, ctx=ctx)

    net(inputs_nd,valid_length_nd).sigmoid()

    target_path = f"./{model_name}_{batchsize}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    net.export(f'{model_name}_{batchsize}/model')
    print("-"*10,f"Download {model_name} complete","-"*10)  

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='lstm' , type=str)
    parser.add_argument('--batchsize',default=8 , type=int)
    parser.add_argument('--seq',default=128 , type=int)


    args = parser.parse_args()

    model_name = args.model
    batchsize = args.batchsize
    seq_length=args.seq


    get_model(model_name,batchsize,seq_length)
    
