import torch
import model
import torch.nn as nn
import torch.optim as optim
import json
with open('data.json','r') as file:
    data = json.load(file)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.model.parameters(), lr=0.1)
for epoch in range(2):
    for qus,ans in data:
        out,sof = model.model(qus.split(),["start","end"],len(ans.split()))
        lables = [model.vocab[ans.split()[i]] for i in range(len(out))]
        loss = criterion(torch.stack(sof),torch.tensor(lables))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')
        print(torch.tensor(lables))

torch.save(model.state_dict,'/result')
