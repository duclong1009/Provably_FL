import matplotlib.pyplot as plt
import json
import numpy as np

data_path = "cifar10_cnum100_pareto"

with open(f"/home/oem/Projects/provably_fl/fedtask/{data_path}/data.json", "r") as f:
    data = json.load(f)

train_arr = np.zeros((100,10))
valid_arr = np.zeros((100,10))
for i, client_name in enumerate(data["client_names"]):
    list_data_idx = data[client_name]
    for label in list_data_idx["dtrain"]['y']:
        train_arr[i,label] += 1
    for label in list_data_idx["dvalid"]['y']:
        valid_arr[i,label] += 1

plt.bar(range(10), valid_arr.sum(0))
plt.xticks(range(10))
plt.xlabel("Class")
plt.ylabel("Number samples")
import os
saved_folder_path = f"plot/image/{data_path}"
if not os.path.exists(saved_folder_path):
    os.makedirs(saved_folder_path)
plt.savefig(f"{saved_folder_path}/valid_distribution.jpg")
plt.clf()

plt.bar(range(10), train_arr.sum(0))
plt.xticks(range(10))
plt.xlabel("Class")
plt.ylabel("Number samples")
import os
saved_folder_path = f"plot/image/{data_path}"
if not os.path.exists(saved_folder_path):
    os.makedirs(saved_folder_path)
plt.savefig(f"{saved_folder_path}/train_distribution.jpg")
plt.clf()