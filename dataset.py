import torch
import pandas as pd
from category_encoders import OrdinalEncoder, CountEncoder

def add_one(n):
    return n + 1

class FMDataset(torch.utils.data.Dataset):
    def __init__(self, df_train: pd.DataFrame,  df_test: pd.DataFrame, use_columns, label_column='Click'):
        encoder = OrdinalEncoder(cols=use_columns, handle_unknown='impute').fit(df_train)
        df_train_X = encoder.transform(df_train).astype('int64')
        df_test_X  = encoder.transform(df_test).astype('int64')

        self.train_X = torch.from_numpy(df_train_X[use_columns].values).long()
        self.train_y = df_train[label_column].values
        self.test_X = torch.from_numpy(df_test_X[use_columns].values).long()
        self.test_y = df_test[label_column].values

        field_dims = list(df_train_X[use_columns].max())
        self.field_dims = list(map(add_one, field_dims))

        self.data_num = self.train_X.size()[0]

    def get_test(self):
        return self.test_X, self.test_y
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.train_X[idx]
        out_label = self.train_y[idx]
        return out_data, out_label