import torch
import torch.optim as optim
import json
with open('data.txt','r') as file:
    data = json.load(file)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
for epoch in range(2):
    for qus,ans in data:
        out,sof = model(qus.split(),["start","end"])
        lables = [vocab[ans.split()[i]] for i in range(len(out))]
        loss = criterion(torch.stack(sof),torch.tensor(lables))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')
        print(torch.tensor(lables))

torch.save(model.state_dict,'/result')
