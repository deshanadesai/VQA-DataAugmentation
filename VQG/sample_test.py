import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import preprocess_get_model
import sys
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
import skimage.io as io
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import T_Att, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image
import pandas as pd
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image    
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    #Load vocab_list for uniskip
    vocab_list = pd.read_csv(args.vocablist_path, header=None)
    vocab_list = vocab_list.values.tolist()[0]
    
    #Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, args.data_path, vocab, 
                            transform, args.batch_size,
                            shuffle=True, num_workers=args.num_workers) 

    # Build the models
    im_encoder = preprocess_get_model.model()
    attention = T_Att()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        im_encoder.cuda()
        attention.cuda()
        decoder.cuda()

    attention.load_state_dict(torch.load(args.attention_path)) 
    decoder.load_state_dict(torch.load(args.decoder_path))

    for i, (images, captions, cap_lengths, qa, qa_lengths, vocab_words) in enumerate(data_loader):
            
    
         images = to_var(images, volatile=True)
         captions = to_var(captions)
        
         img_embeddings = im_encoder(images) 
         uniskip = UniSkip('/Users/tushar/Downloads/code/data/skip-thoughts', vocab_list)
         cap_embeddings = uniskip(captions, cap_lengths)
         cap_embeddings = cap_embeddings.data
         img_embeddings = img_embeddings.data
         ctx_vec = attention(img_embeddings,cap_embeddings)
         outputs = decoder.sample(ctx_vec)
         output_ids = outputs.cpu().data.numpy()

         predicted_q = []
         predicted_a = []
         sample = []
         flag = -1
         for word_id in output_ids:
            word = vocab.idx2word[word_id]
            sample.append(word)
         #    if word == '<end>':
         #        if flag == -1:
         #            predicted_q = sample
         #            sample = []
         #            flag = 0
         #        else:
         #            predicted_a = sample

         # predicted_q = ' '.join(predicted_q[1:])
         # predicted_a = ' '.join(predicted_a[1:])
         sample = ' '.join(sample)
         # print("predicted q was : " + predicted_q)
         print("predicted was : " + sample)
         #plt.imshow(images.data.numpy())
         break

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--vocablist_path', type=str, default='./data/vocab_list.csv',
                        help='path for vocab list file')
    parser.add_argument('--image_dir', type=str, default='/Users/tushar/Downloads/train2014' ,
                        help='dir for images')
    parser.add_argument('--caption_path', type=str,
                        default='./annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--data_path', type=str, default='./dataset.csv' ,
                        help='directory for preprocessed dataset (csv file containing im_id,qid,q,a,captions)')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--attention_path', type=str, default='./models/attention-5-1.pkl',
                        help='path for trained attention')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-1.pkl',
                        help='path for trained decoder')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)