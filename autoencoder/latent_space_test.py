import os
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchsummary import summary

from autoencoder.AutoEncoder import AutoencoderNet
from autoencoder.AutoEncoderWrapper import AutoEncoderWrapper
from autoencoder.seed_pytorch import set_pytorch_seed


def mean_latent_for_class(model, data_loader, class_id, device):
    _size = 0
    latent = torch.zeros((1, code_dim), device=device)
    ml.network.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            mask = (target == class_id)
            if not mask.any():
                continue
            filtered_data = data[mask].to(device)
            encoded = model.encode(filtered_data)
            latent += encoded.sum(dim=0, keepdim=True)
            _size += filtered_data.size(0);
        latent = latent / _size
    return latent

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = "../"
datasets_path = "../datasets/"
n_epochs = 7
batch_size_train = 24
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
noise_factor = 0.2



test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(datasets_path, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=False, **data_loader_kwargs)
code_dim = 12
network = AutoencoderNet(code_dim)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,  weight_decay=1e-05)

model_file_name = f"data/out/autoencoder_0.2_c{code_dim}_model.pth"

ml = AutoEncoderWrapper(network, optimizer, base_path, device)
ml.summary((1,28,28))

print(AutoEncoderWrapper.get_model_path(base_path,model_file_name))
if os.path.exists(AutoEncoderWrapper.get_model_path(base_path,model_file_name)):
    print("Loading model from file:", model_file_name)
    ml = AutoEncoderWrapper.load_model(base_path, model_file_name, network, optimizer=optimizer, device=device)
else:
    print("NO network to be loaded")
    exit(12)


print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class_id = 5
mean = mean_latent_for_class(ml, test_loader, class_id, device)

with torch.no_grad():
  network.to(device)
  output = ml.decode(mean)
output_data = output.data.cpu()
print(output_data.shape)
fig = plt.figure()
plt.imshow(output.data[0][0].cpu(), cmap='gray', interpolation='none')
plt.title("Result:) ")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig(f"../data/out/autoencoder/latent_{class_id}_{code_dim}_e{n_epochs}.png")
plt.close()