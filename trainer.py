from dataloader import Dataset, get_loader
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from logger import setup_custom_logger
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision.models as models
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn


def list_all_files(path):
    files = os.listdir(path)
    return [os.path.join(path, i) for i in files]


class PneumoniaTrainer():
    def __init__(
        self,
        data_root_dir: str,
        inference_result_folder: str,
        exp_name: str,
        img_size: int = 256,
        device: str = 0,
        num_workers: int = 4,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 8,
        network: str = "resnet101",
        is_RandomResizedCrop: bool = False,
        is_RandomRotation: bool = False,
        is_RandomHorizontalFlip: bool = False,
        is_GaussianBlur: bool = False,
        is_RandomPerspective: bool = False,
        is_CenterCrop: bool = False,
        
        patience: bool = 10,

        # for inference
        model_weight: str = None,
    ):
        self.logger = setup_custom_logger(__name__, exp_name)
        self.data_root_dir = data_root_dir
        self.img_size = img_size
        self.device = device
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.network = network
        self.is_RandomRotation = is_RandomRotation
        self.is_RandomHorizontalFlip = is_RandomHorizontalFlip
        self.is_GaussianBlur = is_GaussianBlur
        self.is_RandomPerspective = is_RandomPerspective
        self.is_CenterCrop = is_CenterCrop
        
        self.patience = patience
        self.model_weight = model_weight
        
        if self.device != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.device}")
        else:
            self.device = 'cpu'

        normal_folder_name = "NORMAL"
        defect_folder_name = "PNEUMONIA"
        
        self.train_normal_path = os.path.join(self.data_root_dir, 'train', normal_folder_name)
        self.train_defect_path = os.path.join(self.data_root_dir, 'train', defect_folder_name)
        self.val_normal_path = os.path.join(self.data_root_dir, 'val', normal_folder_name)
        self.val_defect_path = os.path.join(self.data_root_dir, 'val', defect_folder_name)    
        self.test_normal_path = os.path.join(self.data_root_dir, 'test', normal_folder_name)
        self.test_defect_path = os.path.join(self.data_root_dir, 'test', defect_folder_name)

        self.train_normal_files = list_all_files(self.train_normal_path)
        self.train_defect_files = list_all_files(self.train_defect_path)
        self.val_normal_files = list_all_files(self.val_normal_path)
        self.val_defect_files = list_all_files(self.val_defect_path)
        self.test_normal_files = list_all_files(self.test_normal_path)
        self.test_defect_files = list_all_files(self.test_defect_path)

        if self.network=='resnet18':
            self.model = models.resnet18(pretrained = True)
        elif self.network=='resnet50':
            self.model = models.resnet50(pretrained = True)
        elif self.network=='resnet101':
            self.model = models.resnet101(pretrained = True)

        # change fully connected layer
        # TODO: Fix latest layer here
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        self.model.to(self.device)

        # create checkpoint folder
        # to save model weight
        # to plot training loss and acc trend
        # to plot eval loss and acc trend
        self.exp_name = exp_name
        self.inference_result_folder = os.path.join(inference_result_folder, self.exp_name)
        os.makedirs(self.inference_result_folder, exist_ok=True)

    def plot_trend(self, arr, y_axis_label, save_file_name):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(range(1, len(arr)+1), arr)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(y_axis_label)
        plt.title(save_file_name)
        plt.show()
        plt.savefig(os.path.join(self.inference_result_folder, save_file_name))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_file_name):
        cls = ['Normal', 'Pneumonia']
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cls)
        disp.plot()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_result_folder, save_file_name))
        plt.close()
    
    def training(self):
        self.train_loss, self.valid_loss = [], []
        self.train_acc, self.valid_acc = [], []
        self.train_f1, self.valid_f1 = [], []
        self.lr_record = []

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
        # learning rate scheduler
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1) 
        
        criterion = nn.CrossEntropyLoss()

        trainloader = get_loader(**{
            "files": self.train_normal_files+self.train_defect_files,
            "gt": [0]*len(self.train_normal_files)+[1]*len(self.train_defect_files),
            "shuffle": True,
            "img_size": 256,
            "batch_size": self.batch_size,
            "is_train": True,
            "num_workers": self.num_workers,
            "is_RandomRotation": self.is_RandomRotation,
            "is_RandomHorizontalFlip": self.is_RandomHorizontalFlip,
            "is_CenterCrop": self.is_CenterCrop,
            "is_GaussianBlur": self.is_GaussianBlur,
            "is_RandomPerspective": self.is_RandomPerspective,
        })
        
        val_loader = get_loader(**{
            "files": self.val_normal_files+self.val_defect_files,
            "gt": [0]*len(self.val_normal_files)+[1]*len(self.val_defect_files),
            "shuffle": False,
            "img_size": 256,
            "batch_size": self.batch_size,
            "is_train": False,
            "num_workers": self.num_workers,
        })

        best_val_acc = 0
        no_improvement_count = 0

        # Start the training.
        for epoch in range(self.epochs):
            # get current lr
            # cur_lr = optimizer.param_groups[0]["lr"]
            
            ###############
            # training
            ###############
            self.model.train()
            self.logger.info(f'Training, epoch: {epoch}/{self.epochs}')
            train_running_loss = 0.0
            train_running_correct = 0
            counter = 0
            y_true = []
            y_pred = []
    
            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
                counter += 1
                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                # Forward pass.
                outputs = self.model(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
                train_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)

                train_running_correct += (preds == labels).sum().item()

                y_true += labels.detach().cpu().numpy().tolist()
                y_pred += preds.detach().cpu().numpy().tolist()
                
                # Backpropagation
                loss.backward()
                # Update the weights.
                optimizer.step()

            # update lr
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]
            self.lr_record.append(cur_lr)

            # Loss and accuracy for the complete epoch.
            epoch_loss = train_running_loss / counter
            # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
            epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
            epoch_f1 = f1_score(y_true, y_pred)
            self.logger.info(f"train loss: {epoch_loss}, train acc: {epoch_acc:.2f}% ({train_running_correct}/{len(trainloader.dataset)}), lr:{cur_lr}, train f1: {epoch_f1}")

            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            self.train_f1.append(epoch_f1)
    
            ###############
            # evaluation
            ###############
            self.model.eval()
            self.logger.info('Validation')
            valid_running_loss = 0.0
            valid_running_correct = 0
            counter = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    counter += 1
                    image, labels = data
                    image = image.to(self.device)
                    labels = labels.to(self.device)
                    # Forward pass.
                    outputs = self.model(image)
                    # Calculate the loss.
                    loss = criterion(outputs, labels)
                    valid_running_loss += loss.item()
                    # Calculate the accuracy.
                    _, preds = torch.max(outputs.data, 1)
                    valid_running_correct += (preds == labels).sum().item()
                    
                    y_true += labels.detach().cpu().numpy().tolist()
                    y_pred += preds.detach().cpu().numpy().tolist()
                
            # Loss and accuracy for the complete epoch.
            epoch_loss = valid_running_loss / counter
            epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
            epoch_f1 = f1_score(y_true, y_pred)

            self.logger.info(f"val loss: {epoch_loss}, val acc: {epoch_acc:.2f}% ({valid_running_correct}/{len(val_loader.dataset)}), val f1 score: {epoch_f1}")

            self.valid_loss.append(epoch_loss)
            self.valid_acc.append(epoch_acc)
            self.valid_f1.append(epoch_f1)

            # save best
            # and if no improve, early stop
            if best_val_acc < epoch_acc:
                self.logger.info(f"History best: {best_val_acc:.2f} < {epoch_acc:.2f}, saving model.")
                best_val_acc = epoch_acc
                self._dump('best.pt')
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    self.logger.info(f"Eealy stoping at {epoch}.")
                    break
                else:
                    self.logger.info(f"Do not update {no_improvement_count} epoch")

        self.plot_trend(self.train_loss, "train_loss", "train_loss")
        self.plot_trend(self.train_acc, "train_acc", "train_acc")
        self.plot_trend(self.train_f1, "train_f1", "train_f1")
        self.plot_trend(self.valid_loss, "val_loss", "val_loss")
        self.plot_trend(self.valid_acc, "val_acc", "val_acc")
        self.plot_trend(self.valid_f1, "val_f1", "val_f1")
        
        # save lastest
        self._dump('last.pt')

    def _load(self):
        self.logger.info(f"Load model from: {self.model_weight}")
        self.model = torch.load(self.model_weight)
    
    def _dump(self, model_name):
        # save model to inference result folder
        self.model_path = os.path.join(self.inference_result_folder, model_name)
        self.logger.info(f"Saving model.pt to {self.model_path}")
        torch.save(self.model, self.model_path)
    
    def testing(self):
        self.logger.info('testing')
        self.model.eval()

        # parepare testing dataloader
        test_loader = get_loader(**{
            "files": self.test_normal_files+self.test_defect_files,
            "gt": [0]*len(self.test_normal_files)+[1]*len(self.test_defect_files),
            "shuffle": False,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "is_train": False,
            "num_workers": self.num_workers,
        })

        # start inference
        test_running_correct = 0
        counter = 0

        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                counter += 1
                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(image)
                _, preds = torch.max(outputs.data, 1)
                test_running_correct += (preds == labels).sum().item()
                y_true += labels.detach().cpu().numpy().tolist()
                y_pred += preds.detach().cpu().numpy().tolist()
            
        epoch_acc = 100. * (test_running_correct / len(test_loader.dataset))
        epoch_f1 = f1_score(y_true, y_pred)
        self.logger.info(f"testing acc: {epoch_acc}% ({test_running_correct}/{len(test_loader.dataset)}), f1 score: {epoch_f1}")
        
        self.plot_confusion_matrix(y_true, y_pred,"test_confusion_matrix")

        return f"{epoch_acc}% ({test_running_correct}/{len(test_loader.dataset)})"