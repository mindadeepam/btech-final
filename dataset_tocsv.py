import os 
import numpy as np
import pandas as pd

dataset_dir = "/home/lenovo/Desktop/final/dataset"

file_path = []
label = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        file_name = os.path.join(root, file)
        file_path.append(file_name)
        dir_name = file_name.split("/")[-2]
        label.append(dir_name)


data = {"file_path" : file_path, "label":label}

df = pd.DataFrame.from_dict(data)

df.to_csv("dataset.csv")