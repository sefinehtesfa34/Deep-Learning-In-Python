import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


train_data=datasets.FashionMNIST(
    "data",
    train=True,
    download=True,
    transform=ToTensor()
    )
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Linear(784,256)
        self.layer2=nn.Linear(256,128)
        self.layer3=nn.Linear(128,64)
        self.outputLayer=nn.Linear(64,10)
        self.dropout=nn.Drpout(p=0.2)
    def forward(self,x):
    # Flatten the image
        x=x.view(x.shape[0],-1)
        x=self.dropout(F.relu(self.layer1(x)))
        x=self.dropout(F.relu(self.layer2(x)))
        x=self.dropout(F.relu(self.layer3(x)))
        x=F.log_softmax(self.outputLayer(x),dim=1)
        return x
model=Classifier()
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=10
for epoch in range(epochs):
  running_loss=0
  for images,labels in trainloader:
    logps=model(images)
    loss=criterion(logps,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
  print(f"The trianing loss = {running_loss}")
