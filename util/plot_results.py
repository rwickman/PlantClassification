import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

sns.set(style="darkgrid")

train_dict_file = "models/train_dict.json"

with open(train_dict_file) as f:
    train_dict = json.load(f)

def moving_average(x, w=2):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, axs = plt.subplots(2)


axs[0].plot(moving_average(train_dict["train_loss"]))
axs[0].set(xlabel="Train Iter", ylabel="Training Loss")
axs[1].plot(moving_average(train_dict["val_loss"]))
axs[1].set(xlabel="Train Iter", ylabel="Validation Loss")

plt.show()