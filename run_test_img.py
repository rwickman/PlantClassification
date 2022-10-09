import torch
from torchvision import transforms
from PIL import Image
import os

from model import PlantClassifier
from util.config import device 


img_size = 256
img_dir = "test_imgs/"
test_imgs = ["0_basil.jpg", "1_basil.jpg", "2_basil.jpg", "3_basil.jpg", "0_lettuce.jpg", "1_lettuce.jpg"]
test_imgs = [os.path.join(img_dir, test_img) for test_img in test_imgs]

model_file = "models_ugh/checkpoints/model_epoch_100.pkl"
model_classes = ['lactuca_sativa', 'ocimum_basilicum']

# Load the model
model = PlantClassifier(num_classes=2).to(device)
model_dict = torch.load(model_file)
print(model.load_state_dict(model_dict["model"]))
model.eval()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])
test_arr = []
# Open and transform all the images
for test_img in test_imgs:
    test_img = Image.open(test_img).convert("RGB")
    test_img = transforms(test_img).to(device)
    test_arr.append(test_img)

test_arr = torch.stack(test_arr)
print(test_arr.shape)

# Make prediction
pred = torch.nn.functional.softmax(model(test_arr), dim=-1)

print("PREDS", pred, pred.argmax(dim=1))
