import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import c


# CONSTS
RANDOM_SEED = c.RANDOM_SEED
torch.manual_seed(RANDOM_SEED)


class MusicRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0):
		super(MusicRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.embeddings = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def init_hidden(self):
		self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
									 Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

	def forward(self, seq):
		embedded = self.embeddings(seq)
		rnn_out, _ = self.rnn(embedded.view(len(seq), 1, -1))
		output = self.out(rnn_out.view(len(seq), -1))
		return output


