import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
	def __init__(self, input_size, hidden_size_list, output_size, dropout_rate=0.1):
		super(MLP, self).__init__()
		mlp_layers = []

		prev_size = input_size
		for i in hidden_size_list:
			mlp_layers.append(nn.Linear(prev_size, i, bias=True))
			mlp_layers.append(nn.BatchNorm1d(i))
			mlp_layers.append(nn.ReLU())
			mlp_layers.append(nn.Dropout(dropout_rate))
			prev_size = i

		mlp_layers.append(nn.Linear(prev_size, output_size, bias=True))
		self.seq = nn.Sequential(*mlp_layers)

	def forward(self, x):
		return self.seq(x)


