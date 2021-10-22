import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from ..SentenceTransformer import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
import logging
import math
from functools import wraps
import copy
import random


class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
	for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
		old_weight, up_weight = ma_params.data, current_params.data
		ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP for  predictor
class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)


# loss fn
def loss_fn(x, y):
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	return 2 - 2 * (x * y).sum(dim=-1)



class BYOLoss(nn.Module):
	def __init__(self,
				 model: SentenceTransformer,
				 sentence_embedding_dimension: int,
				 moving_average_decay: float):
		super(BYOLoss, self).__init__()
		self.online_encoder = model
		self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension) 
		self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension) 
		self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension) 
		self.target_encoder = copy.deepcopy(self.online_encoder)
		self.target_ema_updater = EMA(moving_average_decay)  

	def update_moving_average(self):
		assert self.target_encoder is not None, 'target encoder has not been created yet'
		update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

		target_sentence_features = copy.deepcopy(sentence_features)
		rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
		online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
		online_pred_one, online_pred_two = self.online_predictor_1(online_pred_one), self.online_predictor_1(online_pred_two)
		online_pred_one, online_pred_two = self.online_predictor_2(online_pred_one), self.online_predictor_2(online_pred_two)
		online_pred_one, online_pred_two = self.online_predictor_3(online_pred_one), self.online_predictor_3(online_pred_two)

		with torch.no_grad():

			target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in target_sentence_features]
			target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']

		loss_one = loss_fn(online_pred_one, target_proj_two.detach())
		loss_two = loss_fn(online_pred_two, target_proj_one.detach())

		loss = loss_one + loss_two

		return loss.mean()

