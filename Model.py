import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# CONSTS
SAVE_EVERY = 20
SEQ_SIZE = 25
RANDOM_SEED = 11
VALIDATION_SIZE = 0.15
LR = 1e-3
N_EPOCHS = 100
NUM_LAYERS, HIDDEN_SIZE = 1, 150
DROPOUT_P = 0
model_type = 'lstm'
use_cuda = torch.cuda.is_available()
torch.manual_seed(RANDOM_SEED)
INPUT = 'data/music.txt'  # Music
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

def some_pass(seq, target, fit=True):
    model.init_hidden() # Zero out the hidden layer
    model.zero_grad()   # Zero out the gradient
    some_loss = 0
    
    for i, c in enumerate(seq):
        output = model(c)
        some_loss += loss_function(output, target[i])
        
    if fit:
        some_loss.backward()
        optimizer.step()
    
    return some_loss.data[0] / len(seq)

# Model
if RESUME:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    checkpoint = torch.load('./checkpoint/' + CHECKPOINT + '.tEP_NUM_SVAED')
    model = checkpoint['model']
    loss = checkpoint['loss']
    v_loss = checkpoint['v_loss']
    losses = checkpoint['losses']
    v_losses = checkpoint['v_losses']
    start_epoch = checkpoint['epoch']
    
else:
    print('==> Building model..')
    in_size, out_size = len(char_idx), len(char_idx)
    model = MusicRNN(in_size, HIDDEN_SIZE, out_size, model_type, NUM_LAYERS)
    loss, v_loss = 0, 0
    losses, v_losses = [], []
    start_epoch = 0

#if use_cuda:
#    net.cuda()
#    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#    cudnn.benchmark = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

# Train
time_since = tic()
for epoch in range(start_epoch, N_EPOCHS):
    # Training
    for i, song_idx in enumerate(train_idxs):
        this_loss = some_pass(*song_to_seq_target(data[song_idx]))
        loss += this_loss
        
        msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
             epoch, (i+1)/len(train_idxs)*100, i, toc(time_since), this_loss)
        sys.stdout.write(msg)
        sys.stdout.flush()
    print()
    losses.append(loss / len(train_idxs))
        
    # Validation
    for i, song_idx in enumerate(valid_idxs):
        this_loss = some_pass(*song_to_seq_target(data[song_idx]), fit=False)
        v_loss += this_loss
        
        msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
             epoch, (i+1)/len(valid_idxs)*100, i, toc(time_since), this_loss)
        sys.stdout.write(msg)
        sys.stdout.flush()
    print()
    v_losses.append(v_loss / len(valid_idxs))
    
    # Save checkpoint.
    if epoch % SAVE_EVERY == 0 and start_epoch != epoch or epoch == N_EPOCHS - 1:
        print('=======>Saving..')
        state = {
            'model': model.module if use_cuda else model,
            'loss': losses[-1],
            'v_loss': v_losses[-1],
            'losses': losses,
            'v_losses': v_losses,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t%s' % epoch)
        torch.save(state, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)
    
    # Reset loss
    loss, v_loss = 0, 0