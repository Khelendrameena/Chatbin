import torch
import model as mod
d_model = 512
n_head = 8
ffn_hidden = 2048
drop_prob = 0.1
n_layers = 6

def chatbin(t,qus):
  if t == "train":
    state_dict = torch.load('model.pt')
    model = tersformer(d_model,n_head,n_layers,drop_prob,vocab,ffn_hidden)
    model.load_state_dict(state_dict)
    out,sof = model(qus.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
  else:
    out,sof = mod.model(qustion.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
