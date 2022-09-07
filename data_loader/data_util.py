import os
import random
from collections import Counter
from typing import Tuple

from util.config import plant_dataset, train_pct
from data_loader.dataset import ImageDataset

def get_paths_and_class_ids() -> Tuple[list, list]:
    img_paths = []
    class_ids = []
    class_id_to_name = []
    for class_id, img_class in enumerate(os.listdir(plant_dataset)):
        class_path = os.path.join(plant_dataset, img_class)
        class_id_to_name.append(img_class)

        for img_path in os.listdir(class_path):
            img_paths.append(os.path.join(class_path, img_path))
            class_ids.append(class_id)
    
    return img_paths, class_ids, class_id_to_name


def split_dataset(img_paths, class_ids):
    # Shuffle the lists together
    temp = list(zip(img_paths, class_ids))
    random.Random(0).shuffle(temp)
    img_paths, class_ids = zip(*temp)

    # Get split indices
    ds_len = len(img_paths)
    train_end_idx = int(ds_len * train_pct)
    val_end_idx = int(ds_len * ((1.0 - train_pct)/2 + train_pct))

    # Split the dataset
    train_img_paths, train_class_ids = img_paths[:train_end_idx], class_ids[:train_end_idx]
    val_img_paths, val_class_ids = img_paths[train_end_idx:val_end_idx], class_ids[train_end_idx:val_end_idx]
    test_img_paths, test_class_ids = img_paths[val_end_idx:], class_ids[val_end_idx:]

    # Create the datasets
    train_dataset = ImageDataset(train_img_paths, train_class_ids, is_train=True)
    val_dataset = ImageDataset(val_img_paths, val_class_ids)
    test_dataset = ImageDataset(test_img_paths, test_class_ids)

    return train_dataset, val_dataset, test_dataset



def create_class_weights(class_ids):
    # Count and sort the number of examples in each class
    sorted_count = sorted(list(dict(Counter(class_ids)).items()), key=lambda x: x[0])

    # Create the class weights
    class_weights = [len(class_ids)/num_ex for _, num_ex in sorted_count]
    class_weights = [w/sum(class_weights) for w in class_weights]
    
    return class_weights

def load_datasets():
    img_paths, class_ids, class_id_to_name = get_paths_and_class_ids()
    train_dataset, val_dataset, test_dataset = split_dataset(img_paths, class_ids)
    
    return train_dataset, val_dataset, test_dataset, class_id_to_name