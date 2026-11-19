from PIL import Image
import torch
import torchvision.models as models
import os



# Place where model will be saved
os.environ["TORCH_HOME"] = "/opt/project/in"
network = "ResNet50"
#network = "VGG16"
#network = "VT"



if network == "ResNet50":
  weights = models.ResNet50_Weights.DEFAULT
  model = models.resnet50(weights=weights)
elif network == "VGG":
  weights = models.VGG16_Weights.DEFAULT
  model = models.vgg16(weights=weights)
else:
  weights = models.ViT_B_16_Weights.DEFAULT
  model = models.vit_b_16(weights=weights)

model.eval()


preprocess = weights.transforms()
print(preprocess)

img = Image.open("../in/jez.jpg")
batch = preprocess(img).unsqueeze(0)


with torch.no_grad():
  prediction = model(batch).squeeze(0).softmax(0)


class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]

print(f"Rozpoznano: {category_name} ({score * 100:.1f}%)")