import pandas as pd
import numpy as np
import torch

def get_predictions(loader, model, device):
    model.eval() #모든 노드 사용
    saved_preds = []
    true_labels = []


    with torch.no_grad(): # 기울기 트래킹을 하지 않겠다
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            saved_preds += scores.tolist()
            true_labels += y.tolist()

    model.train() # drop out 선별적으로 사
    return saved_preds, true_labels


def get_submission(model, loader, test_ids, device):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            score = model(x)
            prediction = score.float()
            all_preds += prediction.tolist()

    model.train()

    df = pd.DataFrame({
        "ID_code" : test_ids.values,
        "target" : np.array(all_preds)
    })

    df.to_csv("sub.csv", index=False)