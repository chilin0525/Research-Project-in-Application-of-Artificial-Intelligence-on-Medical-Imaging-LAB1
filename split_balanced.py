import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split


defect_files = os.listdir("chest_xray/train/PNEUMONIA")
defect_files = [f"chest_xray/train/PNEUMONIA/{i}" for i in defect_files]
tmp = os.listdir("chest_xray/val/PNEUMONIA")
defect_files += [f"chest_xray/val/PNEUMONIA/{i}" for i in tmp]

normal_files = os.listdir("chest_xray/train/NORMAL")
normal_files = [f"chest_xray/train/NORMAL/{i}" for i in normal_files]
tmp = os.listdir("chest_xray/val/NORMAL")
normal_files += [f"chest_xray/val/NORMAL/{i}" for i in tmp]

seed = 1
random.seed(seed)

N = min(len(defect_files), len(normal_files))

updated_trin_normal_files = random.sample(normal_files, N)
updated_trin_defect_files = random.sample(defect_files, N)

updated_trin_normal_files, updated_val_normal_files = train_test_split(updated_trin_normal_files, random_state=seed)
updated_trin_defect_files, updated_val_defect_files = train_test_split(updated_trin_defect_files, random_state=seed)

os.makedirs("custom_chest_xray_balance", exist_ok=True)
os.makedirs("custom_chest_xray_balance/train", exist_ok=True)
os.makedirs("custom_chest_xray_balance/train/NORMAL", exist_ok=True)
os.makedirs("custom_chest_xray_balance/train/PNEUMONIA", exist_ok=True)
os.makedirs("custom_chest_xray_balance/val", exist_ok=True)
os.makedirs("custom_chest_xray_balance/val/NORMAL", exist_ok=True)
os.makedirs("custom_chest_xray_balance/val/PNEUMONIA", exist_ok=True)


for i in updated_trin_normal_files:
    # print(i.split('/')[-1])
    # exit(0)
    os.system(f"cp {i} custom_chest_xray_balance/train/NORMAL/{i.split('/')[-1]}")

for i in updated_trin_defect_files:
    os.system(f"cp {i} custom_chest_xray_balance/train/PNEUMONIA/{i.split('/')[-1]}")

for i in updated_val_normal_files:
    os.system(f"cp {i} custom_chest_xray_balance/val/NORMAL/{i.split('/')[-1]}")

for i in updated_val_defect_files:
    os.system(f"cp {i} custom_chest_xray_balance/val/PNEUMONIA/{i.split('/')[-1]}")
