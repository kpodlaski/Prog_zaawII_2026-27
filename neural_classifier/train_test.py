'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import os

import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_classifier.cnn_model import CNNNet
from neural_classifier.dense_model import DenseNet
from neural_classifier.ml_wrapper import ML_Wrapper

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = "../"
n_epochs = 15
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True, **data_loader_kwargs)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)

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
    print("Start training")
    for epoch in range(1, n_epochs + 1):
      ml.train(epoch, train_loader, test_loader)
    ml.save_model("cnn_network")

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
              output.data.max(1, keepdim=True)[1][e].item()))
            plt.xticks([])
            plt.yticks([])
plt.savefig(base_path+"/data/out/cnn_network_bad_examples.png")
#fig.show()