import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import get_data
from utils import get_predictions
from torch.utils.data import DataLoader


class NN(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()

        # self.net = nn.Sequential(     # 각 layer를 순차적으로 지날 때 유용한
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size, 50),
        #     nn.ReLU(inplace=True), # 말 그대로 output을 생성하지 않고 대체하겠
        #     nn.Linear(50,1)
        # )
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(input_size*hidden_dim, 1)
    def forward(self, x):
        # return torch.sigmoid(self.net(x)).view(-1)
        Batch_size = x.shape[0]
        x = self.bn(x)
        x = x.view(-1,1) # (bath*200,1) (batch,200)
        x = F.relu(self.fc1(x)).reshape(Batch_size,-1) # contiguous and non-contiguous tensor
        return torch.sigmoid(self.fc2(x)).view(-1)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=200,hidden_dim=20).to(DEVICE) # var 0 ~ 199
optimizer = optim.Adam(model.parameters(), lr = 2e-3 , weight_decay=1e-4)
loss_fn = nn.BCELoss() #binary class cross entropy
train_ds, val_ds, test_ds, test_ids = get_data()


train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)


for epoch in range(10):
    probability, true = get_predictions(val_loader, model, device=DEVICE)
    print(f"VALIDation ROC : {metrics.roc_auc_score(true, probability)}")
    # data, target = next(iter(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        scores = model(data)


        loss = loss_fn(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

from utils import get_submission
get_submission(model, test_loader, test_ids, DEVICE)