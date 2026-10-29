'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

base_path = "../"

batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True)


for x,y in train_loader:
    print(x.shape, y.shape)
    for i in range (1):
        print(x[i])
        print(y[i])
        plt.imshow(x[i][0], cmap='gray')
        #plt.show()
        plt.savefig(f'{base_path}/data/out/mnist_{y[i]}.png')
    break










