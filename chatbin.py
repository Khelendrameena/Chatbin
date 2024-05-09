import torch
import model as mod
d_model = 512
n_head = 8
ffn_hidden = 2048
drop_prob = 0.1
n_layers = 6

def chatbin(t,qus,q_len):
  if t == "train":
    state_dict = torch.load('model.pt')
    model = mod.model
    model.load_state_dict(state_dict)
    out,sof = mod.model(qus.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
  else:
    out,sof = mod.model(qus.split(),["start","end"],q_len)
    ans = ' '.join(out)
    print(ans)
