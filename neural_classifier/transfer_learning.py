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
        target_transform=lambda y: y - 1, # zamiast 1-27 na 0-26
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.transpose(x, 1, 2)),
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
            torchvision.transforms.Lambda(lambda x: torch.transpose(x, 1, 2)),
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
ml.network.to(device)
ml.summary((1,28,28))


for epoch in range(1, n_epochs + 1):
    ml.train(epoch, train_loader, test_loader)

ml.save_model("cnn_eminst_network")

print("Test on training set:")
cm = ml.test(train_loader, create_confusion=True)
print("Train Confusion Mattrix")
print(cm)
print("Test on test set:")
cm = ml.test(test_loader, create_confusion=True)
print("Test Confusion Mattrix")
print(cm)


ml.training_plot()


print("Sample results:")
found_results = 0
#examples = enumerate(test_loader)
fig = plt.figure()
for example_data, example_targets in test_loader :
    if found_results>=6:
        break
    with torch.no_grad():
      example_data =example_data.to(device)
      network.to(device)
      output = ml.network(example_data)

    example_data =example_data.cpu()
    for e in range(len(example_data)) :
        if found_results>=6:
            break
        pred = output.data.max(1, keepdim=True)[1][e].item()
        label = example_targets[e].item()
        if pred != label:
            i = found_results
            found_results += 1
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[e][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(
              train_loader.dataset.classes[ 1+ output.data.max(1, keepdim=True)[1][e].item()]))
            plt.xticks([])
            plt.yticks([])
plt.savefig(base_path+"/data/out/cnn_emnist_network_bad_examples.png")
#fig.show()