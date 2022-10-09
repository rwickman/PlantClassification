import json
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
# from mixup import mixup, mixup_loss

from util.config import device
from model import PlantClassifier
from data_loader.data_util import load_datasets, create_class_weights
from data_loader.dataset import ImageDataset




class Trainer:
    def __init__(self, args):
        self.args = args
        # Create the datasets
        print("Loading data...")
        train_ds, val_ds, test_ds, class_id_to_name = load_datasets()
        print(class_id_to_name)

        # Create the model
        self.model = PlantClassifier(num_classes=len(class_id_to_name)).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Computed based on lr goes to lr_min over epoch updates
        lr_gamma = (self.args.lr_min/self.args.lr) ** (1/self.args.epochs)
        print(lr_gamma)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lr_gamma)


        self.train_dataloader = DataLoader(
            train_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = True,
            shuffle=True)
        
        self.val_data_loader = DataLoader(
            val_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = False,
            shuffle=True)

        self.test_data_loader = DataLoader(
            test_ds,
            self.args.batch_size,
            num_workers = 16,
            prefetch_factor=4,
            drop_last = False,
            shuffle=False) 

        # Create the weighted loss function, where weight accounts for class imbalance
        class_weights = create_class_weights(train_ds.class_ids)
        self.loss_fn = nn.CrossEntropyLoss(torch.tensor(class_weights).to(device))
        print(class_weights)

        #self.loss_fn = nn.CrossEntropyLoss()
        
        # Create training dict to store training results
        self.train_dict = {
            "train_loss" : [],
            "val_loss" : [],
            "val_acc" : [],
            "val_f1_score": []
        }
        self.train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        self.model_file = os.path.join(self.args.save_dir, "model.pkl")

        # Create the model directory if it doesn't exist
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            os.mkdir(os.path.join(self.args.save_dir, "checkpoints"))


        if self.args.load:
            self.load()



    def save(self, epoch):
        model_dict = {
            "model" : self.model.state_dict(),
            "optim" : self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        
        self.model_file = os.path.join(self.args.save_dir, f"checkpoints/model_epoch_{len(self.train_dict['train_loss'])}.pkl")
        
        torch.save(model_dict, self.model_file)
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)
    
    def load(self):
        self.model_file = self.args.load
        print("Loading model checkpoint ", self.model_file)

        model_dict = torch.load(self.model_file)
        self.model.load_state_dict(model_dict["model"])
        self.optim.load_state_dict(model_dict["optim"])
        self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])

        with open(self.train_dict_file) as f:
            self.train_dict = json.load(f)

    def _evaluate(self, data_loader):
        self.model.eval()
        val_acc = 0.0
        total_val_loss = 0
        num_exs = 0
        num_correct = 0
        total_preds = []
        total_class_ids = []
        for batch in tqdm(data_loader):
            imgs, class_ids = batch[0].to(device), batch[1].to(device)
            
            preds = self.model(imgs)

            val_loss = self.loss_fn(preds, class_ids)
            total_val_loss += val_loss.item() * imgs.shape[0]
            preds = torch.nn.functional.softmax(preds, dim=-1)
            
            num_correct += (preds.argmax(dim=1) == class_ids).sum()

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 8)
            for i in range(8):
                axs[i].imshow(imgs[i].detach().cpu().permute(1,2,0))
            print("PREDS", preds[:8])
            print("PREDS", preds.argmax(dim=1).tolist()[:8])
            print("CLASSES", class_ids[:8])
            plt.show()


            # Add the preds and class_ids in the batch
            total_preds += preds.argmax(dim=1).tolist()
            total_class_ids += class_ids.tolist()
            

        avg_eval_loss = total_val_loss / len(total_preds)
        eval_acc = num_correct / len(total_preds)
        eval_f1_score = f1_score(total_class_ids, total_preds, average="weighted")
        

        self.model.train()
        return avg_eval_loss, eval_acc.item(), eval_f1_score.item()


    def test(self):
        test_loss, test_acc, test_f1_score = self._evaluate(self.test_data_loader)
        print("Test Loss:", test_loss)
        print("Test Accuracy: {:.3%}".format(test_acc))
        print("Test F1 Score:", test_f1_score)

        return test_loss, test_acc, test_f1_score

    def validate(self):
        print("Testing over validation set...")
        val_loss, val_acc, val_f1_score = self._evaluate(self.val_data_loader)
        print("Validation Loss:", val_loss)
        print("Validation Accuracy: {:.3%}".format(val_acc))
        print("Validation F1 Score:", val_f1_score)

        return val_loss, val_acc, val_f1_score


    def train(self):
        print("Training model")
        # TODO: Add mixup
        for epoch in range(self.args.epochs):
            total_train_loss = 0
            print(f"Training epoch {epoch}...")
            for batch in tqdm(self.train_dataloader):
                imgs, class_ids = batch[0].to(device), batch[1].to(device)
  
                
                # Perform mixup
                #imgs, y_a, y_b, lam = mixup(imgs, class_ids, self.args.mixup_alpha)
        
                preds = self.model(imgs)
                self.optim.zero_grad()

                # Use mixup loss function
                #loss = mixup_loss(self.loss_fn, preds, y_a, y_b, lam)
                loss = self.loss_fn(preds, class_ids)
                
                loss.backward()
                self.optim.step()
                total_train_loss += loss.item() * imgs.shape[0]
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(1, 8)
                # for i in range(8):
                #     axs[i].imshow(imgs[i].detach().cpu().permute(1,2,0))
                # plt.show()
            self.lr_scheduler.step()
            print(self.optim.param_groups[0]["lr"])

            # Save the average training loss
            avg_train_loss = total_train_loss / (len(self.train_dataloader) * self.args.batch_size)
            self.train_dict["train_loss"].append(avg_train_loss)
            
            # Test the model on the validation set
            val_loss, val_acc, val_f1_score = self.validate()
            self.train_dict["val_loss"].append(val_loss)
            self.train_dict["val_acc"].append(val_acc)
            self.train_dict["val_f1_score"].append(val_f1_score)

            if epoch % self.args.save_iter == 0 or epoch + 1 == self.args.epochs:
                self.save(epoch)
