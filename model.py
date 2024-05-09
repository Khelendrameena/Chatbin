import math
import torch
import torch.nn as nn
import json
import torch.optim as optim

#position encoding
class position_1:
		def __init__(self,d_model,vocab):
				self.d_model = d_model
				self.vocab = vocab
				self.sin = math.sin
				self.cos = math.cos
							
		def forward(self,x):
			fun = [self.sin,self.cos]
			inp_pos = [self.vocab[word] for word in x]
			pos_enc = [[fun[abs(int(math.sin(math.pi/2*a)))](b/10000**(2*a/self.d_model)) for a in range(self.d_model)]for b in inp_pos]
			return torch.tensor(pos_enc)
			   
def position(d_model,vocab):
	pos = position_1(d_model,vocab)
	return pos.forward
#end

#multiheadattention class
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.multiheadattention = nn.MultiheadAttention(d_model,n_head)
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

    def forward(self,q,k,v):
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        out = self.multiheadattention(q,k,v)
        return out[0]
#end

#layer normalzation class

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
#end

#feed forward nural network class

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
#end

 #encoderlayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
 #end

 #encoder

class Encoder(nn.Module):
    def __init__(self,d_model,n_head,vocab,n_layers,drop_prob,ffn_hidden):
        super().__init__()
        self.pos = position(d_model,vocab)
        self.em = nn.Embedding(len(vocab),d_model)
        self.vocab = vocab
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for i in range(n_layers)])

    def forward(self,x):
        pos = self.pos(x)
        out = self.em(torch.tensor([self.vocab[word] for word in x]))
        out = torch.add(out,pos)
        for layer in self.layers:
            out = layer(out)
        return out
#end

#decoderlayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
#end

#decoder
class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,vocab,n_head,drop_prob,n_layers):
        super().__init__()
        self.pos = position(d_model,vocab)
        self.em = nn.Embedding(len(vocab),d_model)
        self.vocab = vocab
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for i in range(n_layers)])

        self.linear = nn.Linear(d_model,len(vocab))

    def forward(self, trg, src):
        pos = self.pos(trg)
        trg = self.em(torch.tensor([self.vocab[word] for word in trg]))
        trg = torch.add(trg,pos)
        for layer in self.layers:
            trg = layer(trg, src)

        # pass to LM head
        output = self.linear(trg)
        return output

#end

#trasformer class
class tersformer(nn.Module):
    def __init__(self,d_model,n_head,n_layers,drop_prob,vocab,ffn_hidden):
        super(tersformer,self).__init__()
        self.encoder = Encoder(d_model,n_head,vocab,n_layers,drop_prob,ffn_hidden)
        self.vocab = {value: key for key, value in vocab.items()}

        self.decoder = Decoder(d_model,ffn_hidden,vocab,n_head,drop_prob,n_layers)

    def forward(self,x,sos_token,a_len):
        src = self.encoder(x)
        trg = [sos_token[0]]
        ans = []
        sof = []
        for i in range(a_len):
            out = self.decoder(trg,src)
            out = torch.softmax(out,dim=1)
            sof.append(torch.mul(out[0],5e-9))
            out = torch.argmax(out)
            out = [self.vocab[out.item()]]
            ans.append(out[0])
            if sos_token[1] == out:
            	break
            trg = out

        return ans,sof
#end
with open('vocab.txt','r') as file:
	vocab_data = file.read()

vocab_data_1 = list(set(vocab_data.split()))
vocab = {vocab_data_1[i]:i for i in range(len(vocab_data_1))}

d_model = 512
n_head = 8
ffn_hidden = 2048
drop_prob = 0.1
n_layers = 6

model = tersformer(d_model,n_head,n_layers,drop_prob,vocab,ffn_hidden)

def trainer(lr,epochs):
    with open('data.json','r') as file:
        data = json.load(file)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(epochs):
       for qus,ans in data:
           out,sof = model(qus.split(),["start","end"],len(ans.split()))
           lables = [vocab[ans.split()[i]] for i in range(len(out))]
           loss = criterion(torch.stack(sof),torch.tensor(lables))
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(),'model.pt')
     
def chatbin(t,qus,q_len):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if t == "train": 
    model_train = tersformer(d_model,n_head,n_layers,drop_prob,vocab,ffn_hidden)
    model_train.load_state_dict(torch.load('model.pt'))
    out,sof = model_train(qus.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
  else:
    out,sof = model(qus.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
