'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import os
import torch
from torch import nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_classifier.cnn_model import CNNNet
from neural_classifier.ml_wrapper import ML_Wrapper

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = "../"
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(base_path+'/datasets/', train=True, download=True,
        split='letters', #ma 26 klas może być 'balanced' ale ma 47 klas,
        target_transform=lambda y: y - 1,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1751,), (0.3332,)) # Nowa normalizacja
        ])
    ),
    batch_size=batch_size_train, shuffle=True,**data_loader_kwargs )

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(base_path+'/datasets/', train=False, download=True,
        split='letters', #ma 26 klas może być 'balanced' ale ma 47 klas,
        target_transform=lambda y: y - 1,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1751,), (0.3332,)) # Nowa normalizacja
        ])
    ),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs )

#network = DenseNet()
network = CNNNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)
ml.summary((1,28,28))
fname = "cnn_network"
if os.path.exists('{}/data/out/{}_model.pth'.format(base_path,fname)):
    print("Loading trained model")
    ml = ml.load_model(base_path,'{}_model.pth'.format(fname), network, optimizer,device)
else:
    print("No valid MNIST network")
    exit(1123)


ml.network.fc2 = nn.Linear(in_features=network.fc1.out_features, out_features=26)
layers_to_freeze = [ml.network.conv1, ml.network.conv2, ml.network.fc1]
for layer in layers_to_freeze:
    for param in layer.parameters():
        param.requires_grad = False
ml.optimizer = optim.SGD(ml.network.parameters(), lr=learning_rate,
                      momentum=momentum)
print("Nowa sieć po zamrożeniu warstw")
ml.summary((1,28,28))
ml.network.to(device)

for epoch in range(1, n_epochs + 1):
    ml.train(epoch, train_loader, test_loader)
