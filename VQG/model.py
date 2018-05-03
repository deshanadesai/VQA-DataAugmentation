import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


#Attention model:
class T_Att(nn.Module):
    def __init__(self):
        super(T_Att, self).__init__()
        self.kk=7*7
        self.ann_size=2048
        self.emb_size=512
        self.cap_size=2400
        self.lin1ann=nn.Linear(self.ann_size,self.emb_size)
        self.lin1cap=nn.Linear(self.cap_size,self.emb_size,bias=False)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(self.emb_size,1)
        self.softm=nn.Softmax()

    def forward(self, img_enco_vec,cap_enco_vec):
        s = img_enco_vec.size() 
        n_batch = s[0] # (nbatch,ann_size,k,k) so the input from encoding model should be [2048,7,7]
        #imgenco_vec = img_enco_vec.view(1,s[0] * s[2] * s[2], s[1])  # (1->for saying that the entire batch is one now *fixed value, nbatch*kk,ann_size) kk=k*k
        imgenco_vec = img_enco_vec.view(1,s[0] * s[3] * s[3], s[2])
        imgenco_var = to_var(imgenco_vec)
        imgvec1=self.lin1ann(imgenco_var) #1*(nbatch*kk,emb_size)
        s2=cap_enco_vec.size()
        #(nbatch,cap_size)  so cap_size input from encoding model should be [2400]
        cap_enco_vec=cap_enco_vec.view(1,s2[0],s2[1])
        cap_enco_var=to_var(cap_enco_vec)
        capvec1=self.lin1cap(cap_enco_var)
        capvec1=capvec1.unsqueeze(2).repeat(1,1,self.kk,1).view(1,-1,self.emb_size) #unsqueeze 2 means rowwise, ends with 1*(nbatch*kk,emb_size)
        vecvar=imgvec1+capvec1 #1*(nbatch*kk,emb_size)
        #vecvar=Variable(vec)
        tanhvar=self.relu(vecvar)#1*(nbatch*kk,emb_size)
        #tanhvar=Variable(tanhvec)
        attvar=self.lin2(tanhvar) #1*(nbatch * kk, 1)
        attvar=attvar.view(1,n_batch,self.kk) #1*(nbatch,kk)
        #attvar=Variable(attvec)
        attsoft=self.softm(attvar)#1*(nbatch,kk)
        alphas=attsoft.view(-1)#(nbatch*kk)
        alphasvec=alphas.unsqueeze(1).repeat(1,self.ann_size) #(nbatch*kk,ann_size)
        #print(alphasvec.type())
        #print(imgenco_vec.type())
        alphasvec = alphasvec.data
        ctxs=torch.mul(alphasvec,imgenco_vec)#(1,nbatch*kk,ann_size)
        ctxs=ctxs.view(n_batch,self.kk,self.ann_size) #(nbatch,kk,ann_size)
        ctxs=torch.sum(ctxs,dim=1) #(nbatch,ann_size)
        return ctxs    

#Decoder to generate QA pair:    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_embed = nn.Linear(2048,embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.ctx_embed.weight.data.uniform_(-0.1,0.1)
        self.ctx_embed.bias.data.fill_(0)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def init_hidden(self):
        """Initialize hidden state"""
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
     
        
    def forward(self, ctx_vec, qa, lengths):
        """Decode image feature vectors and generates QA."""
        embeddings = self.embed(qa)
        ctx_var = Variable(ctx_vec)
        ctx_embeddings = self.ctx_embed(ctx_var)
        embeddings = torch.cat((ctx_embeddings.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        #output_scores = softmax(outputs, dim=1)
        return outputs

    def sample(self, ctx_vec, states = None):
        """Samples qa for given context vector."""
        sampled_ids = []
        #embeddings = self.embed(qa)
        ctx_var = Variable(ctx_vec)
        ctx_embeddings = self.ctx_embed(ctx_var)
        inputs = ctx_embeddings.unsqueeze(1)
        for i in range(32): #max qa len with tags
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  

        sampled_ids = torch.cat(sampled_ids, 0)
        #print(sampled_ids.size())       
        return sampled_ids.squeeze()    
    
