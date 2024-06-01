import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1, device='cuda'):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim).to(device)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,))).to(device)

        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]),  dtype=torch.int64, device=device)

        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]),  dtype=torch.long, device=device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        a=torch.sum(self.fc(x), dim=1)
        b=self.bias
        c=a+b
        return c

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, device='cuda'):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim, device=device).to(device)
        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.int64, device=device)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)####
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FM(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1)

# class DeepFM(nn.Module):
#     def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
#         super().__init__()
#         self.linear = FeaturesLinear(field_dims)
#         self.fm = FactorizationMachine(reduce_sum=True)
#         self.embedding = FeaturesEmbedding(field_dims, embed_dim)
#         self.embed_output_dim = len(field_dims) * embed_dim
#         self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         embed_x = self.embedding(x)
#         x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
#         return x.squeeze(1)