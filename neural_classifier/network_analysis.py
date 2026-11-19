import os

import numpy as np
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

mnist_test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)

emnist_test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(base_path+'/datasets/', train=False, download=True,
        split='letters', #ma 26 klas może być 'balanced' ale ma 47 klas,
        target_transform=lambda y: y - 1,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1751,), (0.3332,)) # Nowa normalizacja
        ])
    ),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs )
test_loader = mnist_test_loader
#test_loader = emnist_test_loader
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
    print("There is no network to load")

ml.register_hook('conv1', ml.network.conv1)
ml.register_hook('max2', ml.network.max1)

#get first batch from test set
imgs, labels = next(iter(test_loader))
imgs = imgs.to(device)
print(imgs.shape)
predict = ml.network(imgs)
result = predict.data.max(1, keepdim=True)[1][0].item()

activation_img = ml.activations['conv1'][0].numpy()
img = imgs[0].detach().cpu().numpy()

print(activation_img.shape)
fig = plt.figure()
plt.subplot(4, 4, 1)
plt.tight_layout()


pred_data = predict.data[0].detach().cpu().numpy()
print("winning class:",result, "winning value:",np.exp(pred_data)[result])
print("final answer vector", pred_data)
print("final probabilities", np.exp(pred_data))
plt.imshow(img[0], cmap='gray', interpolation='none')
plt.title(f"Original")
plt.xticks([])
plt.yticks([])
for i in range(10):
      plt.subplot(4, 4, i + 5)
      plt.tight_layout()
      plt.imshow(activation_img[i], cmap='gray', interpolation='none')
      plt.title(f"Act {i}")
      plt.xticks([])
      plt.yticks([])
plt.tight_layout()
fig.savefig(f"../data/out/activations.png")
plt.close()

conv1 = ml.network.conv1.weight.cpu().detach().numpy()
fig, axes = plt.subplots(3, 4)

for i, ax in enumerate(axes.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    if (i>9): break
    im= ax.imshow(conv1[i][0], cmap='gray', interpolation='none')
    ax.set_title(f"Kernel {i}")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.04)
# 4. Label the color bar scale
cbar.set_label('Activation Intensity', fontsize=12, weight='bold', labelpad=15)

fig.savefig(f"../data/out/kernels.png")
plt.close()
