import torch

device = device = "cuda" if torch.cuda.is_available() else "cpu"
train_pct = 0.8
plant_dataset = "/media/data/datasets/plant/classification"


