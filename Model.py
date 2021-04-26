import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
#import torchtext
import time
import math
import random
import pandas as pd
import numpy as np
import glob
import pickle
import os
from music21 import converter, instrument, note, chord

# CONSTS
SAVE_EVERY = 20
SEQ_SIZE = 25
RANDOM_SEED = 11
VALIDATION_SIZE = 0.15
LR = 1e-3
N_EPOCHS = 10
NUM_LAYERS, HIDDEN_SIZE = 1, 150
DROPOUT_P = 0
model_type = 'lstm'
use_cuda = torch.cuda.is_available()
torch.manual_seed(RANDOM_SEED)
RESUME = False
CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(model_type, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)

class MusicRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, model='gru', num_layers=1):
		super(MusicRNN, self).__init__()
		self.model = model
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.embeddings = nn.Embedding(input_size, hidden_size)
		if self.model == 'lstm':
			self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
		elif self.model == 'gru':
			self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
		else:
			raise NotImplementedError
		self.out = nn.Linear(self.hidden_size, self.output_size)
		self.drop = nn.Dropout(p=DROPOUT_P)

	def init_hidden(self):
		if self.model == 'lstm':
			self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
										 Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
		elif self.model == 'gru':
			self.hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))

	def forward(self, seq):
		embeds = self.embeddings(seq.view(1, -1))
		rnn_out, self.hidden = self.rnn(embeds.view(1,1,-1), self.hidden)
		rnn_out = self.drop(rnn_out)
		output = self.out(rnn_out.view(1,-1))
		return output

def __init__():
	return
