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
import Model
import matplotlib.pyplot as plt

# CONSTS
SAVE_EVERY = 5
SEQ_SIZE = 25
RANDOM_SEED = 11
TRAINING_SIZE = 0.75
VALIDATION_SIZE = 0.15
TESTING_SIZE = 0.10
LR = 1e-3
N_EPOCHS = 30
NUM_LAYERS, HIDDEN_SIZE = 1, 150
DROPOUT_P = 0
model_type = 'lstm'
torch.manual_seed(RANDOM_SEED)
CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(model_type, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)

class ModelTrainer():

	def __init__(self, loss_function, data, vocab, model=None):

		# Get dataset
		self.data = data
		self.vocab = vocab

		# Get model parameters
		self.model = model
		self.loss_function = loss_function

		# Training parameters
		self.loss, self.v_loss = None, None
		self.losses, self.v_losses = None, None
		self.epoch = None
		self.checkpoint = 0

	def rand_song_slice(self, song, slice_length):
		start_element_i = math.floor(random.random() * (len(song) - slice_length - 1))
		end_element_i = start_element_i + slice_length + 1
		return song[start_element_i: end_element_i]

	def slice_to_tensor(self, slice):
		out = torch.zeros(len(slice)).long()
		for i,j in enumerate(slice):
			out[i] = self.vocab[j]
		return out

	def song_to_seq_target(self, song):
		a_slice = self.rand_song_slice(song, 50)
		seq = self.slice_to_tensor(a_slice[:-1])
		target = self.slice_to_tensor(a_slice[1:])
		assert(len(seq) == len(target)), 'SEQ AND TARGET MISMATCH'
		return Variable(seq), Variable(target)

	def training_pass(self, seq, target):
		self.model.init_hidden() # Zero out the hidden layer
		self.model.zero_grad()   # Zero out the gradient
		some_loss = 0

		output = []
		for i, c in enumerate(seq):
			output.append(self.model(c))

		output = torch.cat(output)

		some_loss = self.loss_function(output, target)
		some_loss.backward()
		self.optimizer.step()

		return some_loss.data

	def validation_pass(self, seq, target):
		self.model.init_hidden() # Zero out the hidden layer
		self.model.zero_grad()   # Zero out the gradient
		some_loss = 0

		output = []
		for i, c in enumerate(seq):
			output.append(self.model(c))

		output = torch.cat(output)

		some_loss = self.loss_function(output, target)

		return some_loss.data

	def testing_pass(self):
		return

	def split_dataset(self):
		# NOW SPLIT INTO TRAIN/VALIDATION SETS
		data_length = len(self.data)
		train_end_index = math.floor(data_length*TRAINING_SIZE)
		validation_end_index = train_end_index + math.floor(data_length*VALIDATION_SIZE)

		# Shuffle data and split
		indices = list(range(data_length))
		np.random.seed(RANDOM_SEED)
		np.random.shuffle(indices)

		# Split Data
		train_indices = indices[:train_end_index]
		validation_indices = indices[train_end_index:validation_end_index]
		test_indices = indices[validation_end_index:]

		return train_indices, validation_indices, test_indices

	def plot_loss(self):
		plt.rc('font', size=12)          # controls default text sizes
		plt.rc('axes', titlesize=12)     # fontsize of the axes title
		plt.rc('axes', labelsize=0)      # fontsize of the x and y labels
		plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
		plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
		plt.rc('legend', fontsize=12)    # legend fontsize
		plt.rc('figure', titlesize=12)   # fontsize of the figure title
		plt.plot(self.losses, label='Training Loss')
		plt.plot(self.v_losses, label='Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('Loss per Epoch')
		plt.legend()
		plt.show()

	def save_checkpoint(self, epoch):
		if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS - 1:
			print('=======>Saving..')
			state = {
				'model': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'loss': self.losses[-1],
				'v_loss': self.v_losses[-1],
				'losses': self.losses,
				'v_losses': self.v_losses,
				'epoch': epoch,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')

			torch.save(state, './checkpoint/' + CHECKPOINT + '.pt')

	def setup_training(self):
		if self.model is None:
			print('==> Resuming from checkpoint...')
			assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

			checkpoint = torch.load('./checkpoint/' + CHECKPOINT + '.pt')
			self.model = Model.MusicRNN()
			self.optimizer = torch.optim.Adam()
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.loss = checkpoint['loss']
			self.v_loss = checkpoint['v_loss']
			self.losses = checkpoint['losses']
			self.v_losses = checkpoint['v_losses']
			self.epoch = checkpoint['epoch']

		else:
			print('==> Starting Fresh...')
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
			self.loss, self.v_loss = 0, 0
			self.losses, self.v_losses = [], []
			self.epoch = 0

	def start_training(self):

		self.setup_training()

		train_indices, validation_indices, test_indices = self.split_dataset()

		# Train
		for epoch in range(self.epoch, N_EPOCHS):
			# Training
			for i, song_index in enumerate(train_indices):
				this_loss = self.training_pass(*self.song_to_seq_target(self.data[song_index]))
				self.loss += this_loss

				msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(
					epoch, (i+1)/len(train_indices)*100, i, this_loss)
				sys.stdout.write(msg)
				sys.stdout.flush()
			print()
			self.losses.append(self.loss / len(self.data))
			print('Train Accuracy: ' + str(self.computeAccuracy(train_indices)))

			for i, song_index in enumerate(validation_indices):
				this_loss = self.validation_pass(*self.song_to_seq_target(self.data[song_index]))
				self.v_loss += this_loss

				msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(
						epoch, (i+1)/len(validation_indices)*100, i, this_loss)
				sys.stdout.write(msg)
				sys.stdout.flush()
			print()
			self.v_losses.append(self.v_loss / len(validation_indices))
			print('Validation Accuracy: ' + str(self.computeAccuracy(validation_indices)))
			self.save_checkpoint(epoch)

			# Reset loss
			self.loss, self.v_loss = 0, 0
	
	def computeAccuracy(self,song_indices):
		with torch.no_grad():
			correct_pred, num_examples = 0, 0
			for i,song_index in enumerate(song_indices):
				notes,targets = self.song_to_seq_target(self.data[song_index])
				for i,c in enumerate(notes):
					logits = self.model(c)
					_, predicted_label = torch.max(logits, 1)
					if (predicted_label == targets[i]):
						correct_pred += 1
				num_examples += targets.size(0)
			print(correct_pred)
			print(num_examples)
			return (correct_pred/num_examples) * 100
	