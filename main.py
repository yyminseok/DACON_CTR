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
from model import FM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn.functional as F
from dataset import FMDataset

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
test_data = pd.read_csv(test_path)


use_columns = list(test_data.drop(columns=['ID', 'Click']).index)
label_column = 'Click'

dataset = FMDataset(train_data, test_data, use_columns, label_column)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)





#Select x,y
train_x = train_data.drop(columns=['ID', 'Click'])
train_y = train_data['Click']

test_x = test_data.drop(columns=['ID'])



#Fill NaN
# for col in tqdm(train_x.columns):
#     if train_x[col].isnull().sum() != 0:
#         train_x[col].fillna(0, inplace=True)
#         test_x[col].fillna(0, inplace=True)
train_x.fillna(0, inplace=True)
test_x.fillna(0, inplace=True)

label_encoder = LabelEncoder()
min_max_scaler = MinMaxScaler()

# object_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
# float_target=list(train_x.dtypes[train_x.dtypes == "float"].index)

categorical_columns = train_x.select_dtypes(include=['object', 'category']).columns
continuous_columns = train_x.select_dtypes(include=[np.number]).columns

# 범주형 데이터 문자열 변환 및 인코딩
for col in categorical_columns:
    train_x[col] = train_x[col].astype(str)
    train_x[col] = LabelEncoder().fit_transform(train_x[col])
    test_x[col] = test_x[col].astype(str)
    test_x[col] = LabelEncoder().fit_transform(test_x[col])

# 연속형 데이터 스케일링
scaler = MinMaxScaler()
train_x[continuous_columns] = scaler.fit_transform(train_x[continuous_columns])
test_x[continuous_columns] = scaler.fit_transform(test_x[continuous_columns])

# TensorDataset과 DataLoader 준비
train_x_tensor = torch.tensor(train_x.values, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32).unsqueeze(1)


dataset = TensorDataset(train_x_tensor, train_y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


#모델 정의 (DeepFM)
feature_sizes = [len(np.unique(train_x[col])) for col in train_x.columns]
model = FM(feature_sizes=feature_sizes, embedding_size=4, hidden_dims=[32, 32], num_classes=1, use_cuda=True)

# train_x = train_data.drop(columns=['ID', 'Click'])
# train_y = train_data['Click']

# test_x = test.drop(columns=['ID'])

# features = torch.tensor(train_data.drop(columns=['ID', 'Click']).values, dtype=torch.float32)
# targets = torch.tensor(train_data['Click'].values, dtype=torch.float32).unsqueeze(1)
# dataset = TensorDataset(features, targets)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
model.fit(loader, optimizer)