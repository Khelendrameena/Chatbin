import torch
qus = "hay hello how are you"
d_model = 512
n_head = 8
ffn_hidden = 2048
drop_prob = 0.1
n_layers = 6

state_dict = torch.load('/result')

model = tersformer(d_model,n_head,n_layers,drop_prob,vocab,ffn_hidden)

model.load_state_dict(state_dict)

out,sof = model(qustion.split(),["start","end"])
ans = ' '.join(out)
print(ans)
