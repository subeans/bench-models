import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import time 

ctx = mx.cpu(0)

def get_models(seq_length):
    # pretrained SOTA Transformer
    model_name = 'transformer_en_de_512'
    dataset_name = 'WMT2014'

    transformer_model, src_vocab, tgt_vocab = \
                    nlp.model.get_model(model_name, dataset_name=dataset_name,
                                                pretrained=True, ctx=ctx)

    import translation
    translator = translation.BeamSearchTranslator(
        model=transformer_model,
        beam_size= 5,
        scorer=nlp.model.BeamSearchScorer(alpha=0.6, K=5),
        max_length=seq_length)

    detokenizer = nlp.data.SacreMosesDetokenizer()
    
    # export 
    #target_path = "./transformer"
    #from pathlib import Path
    #Path(target_path).mkdir(parents=True, exist_ok=True)  

    #transformer_model.export('transformer/model')
    #print("-"*10,f"Download transformer complete","-"*10) 


    return transformer_model , translator , detokenizer, src_vocab, tgt_vocab

def _bpe_to_words(sentence, delimiter='@@'):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = ''
    delimiter_len = len(delimiter)
    for subwords in sentence:
        if len(subwords) >= delimiter_len and subwords[-delimiter_len:] == delimiter:
            word += subwords[:-delimiter_len]
        else:
            word += subwords
            words.append(word)
            word = ''
    return words

def translate(translator, src_seq,src_vocab, tgt_vocab, detokenizer , seq_length,ctx):
    src_sentence = src_vocab[src_seq.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)

    src_nd = mx.nd.array(src_npy)
    src_nd = src_nd.reshape((1, -1)).as_in_context(ctx)
    
    src_valid_length = mx.nd.array([src_nd.shape[1]]).as_in_context(ctx)
    
    samples, _, sample_valid_length = \
        translator.translate(src_seq=src_nd, src_valid_length=src_valid_length)
    max_score_sample = samples[:, 0, :].asnumpy()
    sample_valid_length = sample_valid_length[:, 0].asnumpy()
    
    translation_out = []
    for i in range(max_score_sample.shape[0]):
        translation_out.append(
            [tgt_vocab.idx_to_token[ele] for ele in
             max_score_sample[i][1:(sample_valid_length[i] - 1)]])
        
    real_translation_out = [None for _ in range(len(translation_out))]
    for ind, sentence in enumerate(translation_out):
        real_translation_out[ind] = detokenizer(_bpe_to_words(sentence),
                                                return_str=True)
    return real_translation_out   




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='transformer' , type=str)
    parser.add_argument('--seq',default=128 , type=int)
    parser.add_argument('--input',default="subin is a student" , type=str)

    args = parser.parse_args()
    model_name = args.model
    seq_length = args.seq
    src_seq=args.input

    transformer_model, translator , detokenizer, src_vocab, tgt_vocab = get_models(seq_length)
    output_tgt_seq = translate(translator,      
                                src_seq,                      
                                 src_vocab,
                                 tgt_vocab,
                                 detokenizer,
                                seq_length,
                                 ctx)
