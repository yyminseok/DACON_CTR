import torch
import pandas as pd
from category_encoders import OrdinalEncoder, CountEncoder
from tqdm import tqdm

def add_one(n):
    return n + 1

class FMDataset(torch.utils.data.Dataset):
    def __init__(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, use_columns, label_column='Click'):

        #fill NaN
        for col in tqdm(df_train.columns):
            if df_train[col].isnull().sum() != 0:
                df_train[col].fillna(0, inplace=True)
                df_valid[col].fillna(0, inplace=True)
                df_test[col].fillna(0, inplace=True)
        #encoding
        train_y = df_train[label_column]
        train_x = df_train.drop(columns=['ID', 'Click'])
        valid_x = df_valid.drop(columns=['ID', 'Click'])
        test_x = df_test.drop(columns=['ID'])


        encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
        encoder=CountEncoder(cols = encoding_target).fit(train_x, train_y)

        df_train_X = encoder.transform(train_x).astype('int64')
        df_valid_X  = encoder.transform(valid_x).astype('int64')
        df_test_X  = encoder.transform(test_x).astype('int64')

        self.train_X = torch.from_numpy(df_train_X[use_columns].values).long()
        self.train_y = df_train[label_column].values
        self.valid_X = torch.from_numpy(df_valid_X[use_columns].values).long()
        self.valid_y = df_valid[label_column].values
        self.test_X = torch.from_numpy(df_test_X[use_columns].values).long()
    

        field_dims = list(df_train_X[use_columns].max())
        self.field_dims = list(map(add_one, field_dims))

        self.data_num = self.train_X.size()[0]

    def get_valid(self):
        return self.valid_X, self.valid_y
    
    def get_test(self):
        return self.test_X
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.train_X[idx]
        out_label = self.train_y[idx]
        return out_data, out_label