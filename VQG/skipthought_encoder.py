import sys
import os
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
import torch
import string
from torch.autograd import Variable
import get_captions

def embed(vocab):
    dir_st = 'data/skip-thoughts'
    #vocab = get_captions.get_captions_method()
    vocab=vocab.split(' ')
    vocab=[word.strip(string.punctuation) for word in vocab]
    vocab.append('<eos>')
    vocab = ['<bos>'] + vocab
    #print(vocab)
    #print(len(vocab))
    uniskip = UniSkip(dir_st, vocab)
    r = [r for r in range(1, len(vocab)+1)]
    #print(r)
    input = Variable(torch.LongTensor([r])) # <eos> token is optional
    #print(input.size()) # batch_size x seq_len

    #output_seq2vec = uniskip(input, lengths=[4])
    #print(output_seq2vec.size()) # batch_size x 2400

    output_seq2seq = uniskip(input)
    #print(output_seq2seq.size()) # batch_size x seq_len x 2400'''
    return output_seq2seq

if __name__=='__main__':
    main()
