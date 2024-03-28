from trainer import PneumoniaTrainer
from PIL import Image
from datetime import datetime
import torch
import numpy as np
import random
import pandas as pd
import os


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything()

# using current time as file name
exp_name = datetime.now()
exp_name = exp_name.strftime("%Y-%m-%d %H:%M:%S")

args = {
    "exp_name": exp_name,
    "data_root_dir": 'custom_chest_xray_balance',
    "inference_result_folder": "InferenceResult",
    "img_size": 256,
    "device": 0,
    "num_workers": 8,
    "epochs": 300,
    "batch_size": 64,
    "network": "resnet18",
    "lr": 0.001,

    "is_RandomRotation": True,
    "is_RandomHorizontalFlip": True,
    "is_RandomResizedCrop": True,
    "is_GaussianBlur": False,
    "is_RandomPerspective": False,
    "is_CenterCrop": False,

    "patience": 50
}

pneumonia_trainer = PneumoniaTrainer(**args)

pneumonia_trainer.training()
test_acc = pneumonia_trainer.testing()

df = pd.DataFrame({
    "exp_name": [exp_name],
    "args": [args],
    "train_loss": [pneumonia_trainer.train_loss],
    "train_acc": [pneumonia_trainer.train_acc],
    "valid_loss": [pneumonia_trainer.valid_loss],
    "valid_acc": [pneumonia_trainer.valid_acc],
    "test_acc": [test_acc],
    "lr": [pneumonia_trainer.lr_record]
})

exp_record_path = 'ExpRecord.csv'
df.to_csv(exp_record_path, mode='a', index=False, header=not os.path.exists(exp_record_path))