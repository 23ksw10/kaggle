import pandas as pd
import torch
from torch.utils.data import TensorDataset # 학습데이터를 tensor로 묶는다
from torch.utils.data.dataset import random_split
from math import ceil

def get_data():
    train_data = pd.read_csv('train.csv')
    y = train_data["target"]
    X = train_data.drop(["ID_code",'target'], axis = 1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    data_set = TensorDataset(X_tensor, y_tensor) # 왜 쓰냐? 데이터를 미니배치 단위로 처리할 수 있고,
    train_ds, val_ds = random_split(data_set, [int(0.8*len(data_set)), ceil(0.2*len(data_set))])    # 데이터를 무작위로 섞음으로써 학습의 효율성을 향상시킬 수 있다

    test_data = pd.read_csv('test.csv')
    test_ids = test_data["ID_code"]
    X = test_data.drop(["ID_code",], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)  # 왜 쓰냐? 데이터를 미니배치 단위로 처리할 수 있고,

    return train_ds, val_ds, test_ds, test_ids