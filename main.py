import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb
import torch
import torch.optim as optim
from torch import nn
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
from trainer import FMTrainer

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)

# seed_everything(42) # Seed를 42로 고정


def main():
    #dataset 경로
    train_path = '../ctr_data/train.csv'
    test_path='../ctr_data/test.csv'

    #train data 일부만 가져오기
    con = duckdb.connect()
    train_data = con.query(f"""(SELECT *
                            FROM read_csv_auto('{train_path}')
                            WHERE Click = 0
                            ORDER BY random()
                            LIMIT 3000)
                            UNION ALL
                            (SELECT *
                            FROM read_csv_auto('{train_path}')
                            WHERE Click = 1
                            ORDER BY random()
                            LIMIT 3000)""").df()

    con.close()
    test_data = pd.read_csv(test_path)

    # tain, val 분할
    print('tain, val 분할')
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)

    #make dataset
    print('make dataset')
    use_columns = train_data.drop(columns=['ID','Click']).columns.tolist()
    label_column = 'Click'
    dataset = FMDataset(train_data, val_data, test_data, use_columns, label_column)

    #Setting
    print('Setting')
    loss_fn=nn.BCEWithLogitsLoss()
    optimizer='Adam'
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    field_dims = dataset.field_dims
    hypara_dict = {'embed_rank': 10, 'field_dims': field_dims, 'drop_out': 0.25}

    trainer = FMTrainer(loss_fn, optimizer, device, hypara_dict)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    valid_X, valid_y = dataset.get_valid()

    #Train
    print('Train')
    trainer.fit(train_loader, valid_X, valid_y, epochs=40)

    #Predict
    test_X = dataset.get_test()
    pred_probs = trainer.predict(test_X)


if __name__ == '__main__':
    main()