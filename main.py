import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb
import torch
import torch.optim as optim
from torch.utils.data import sampler
from torch.utils.data import DataLoader, TensorDataset
import plotly.express as px
import category_encoders as ce
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DeepFM import DeepFM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed를 42로 고정

#dataset 경로
train_path = '../ctr_data/train.csv'
test_path='../ctr_data/test.csv'

con = duckdb.connect()
train_data = con.query(f"""(SELECT *
                        FROM read_csv_auto('{train_path}')
                        WHERE Click = 0
                        ORDER BY random()
                        LIMIT 35000)
                        UNION ALL
                        (SELECT *
                        FROM read_csv_auto('{train_path}')
                        WHERE Click = 1
                        ORDER BY random()
                        LIMIT 35000)""").df()

con.close()
test = pd.read_csv(test_path)

#Select x,y
train_x = train_data.drop(columns=['ID', 'Click'])
train_y = train_data['Click']

test_x = test.drop(columns=['ID'])

#Fill NaN
for col in tqdm(train_x.columns):
    if train_x[col].isnull().sum() != 0:
        train_x[col].fillna(0, inplace=True)
        test_x[col].fillna(0, inplace=True)

object_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
float_target=list(train_x.dtypes[train_x.dtypes == "float"].index)


# 범주형 특성 인코딩
label_encoders = {}
for cat_col in object_target:
    le = LabelEncoder()
    train_data[cat_col] = le.fit_transform(train_data[cat_col])
    label_encoders[cat_col] = le

# 연속형 특성 스케일링
scaler = MinMaxScaler()
train_data[float_target] = scaler.fit_transform(train_data[float_target])

feature_sizes = [len(le.classes_) for le in label_encoders.values()]


#모델 정의 (DeepFM)
model = DeepFM(feature_sizes=feature_sizes, embedding_size=4, hidden_dims=[32, 32], num_classes=1, use_cuda=True)


train_x = train_data.drop(columns=['ID', 'Click'])
train_y = train_data['Click']

test_x = test.drop(columns=['ID'])

features = torch.tensor(train_data.drop(columns=['ID', 'Click']).values, dtype=torch.float32)
targets = torch.tensor(train_data['Click'].values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(features, targets)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader):
    model.train()
    for features, target in data_loader:
        optimizer.zero_grad()
        output = model(features)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()

train_model(model, loader)