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
from music21 import converter, instrument, note, chord


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

GENRES = ["Blues", "Country", "Indie", "Jazz", "Pop", "Psychedelic Rock", "Rock", "Soul"]

def get_note_str(note, duration):
	return note.nameWithOctave + duration

def get_element_str(el, includeDuration):
	duration = "-" + str(round(float(el.quarterLength), 3)) if includeDuration else ""

	if isinstance(el, note.Note):
		return get_note_str(el, duration)
	elif isinstance(el, note.Rest):
		return el.name + duration
	elif isinstance(el, chord.Chord):
		note_strings = [get_note_str(n, duration) for n in el.notes]
		return " ".join(sorted(note_strings))

def get_dataset(includeDuration = False, byGenre = False, regenerate = False):

	fileString = "musicDataset_includeDuration=" +\
							 str(includeDuration) +\
							 "_byGenre=" +\
							 str(byGenre) +\
							 ".pickle"

	# Check if this dataset is already generated
	if not regenerate:
		try:
			previouslyGenerateDataset = open(fileString, "rb")
			return pickle.load(previouslyGenerateDataset)
		except:
			print("Re/Creating Music Dataset")
	
	# Create the data set by converting all the midi files into string vectors
	
	dataset = {"total": []}
	for genre in GENRES:
		for file in glob.glob("./TrainingData/" + genre + "/*.mid"):

			print("Parsing %s" % file)
			midi = converter.parse(file)

			try:
				s2 = instrument.partitionByInstrument(midi)
				notes_to_parse = s2.parts[0].recurse()
			except:
				notes_to_parse = midi.flat.notes

			# Create the song vector
			song = []
			for el in notes_to_parse:
				if isinstance(el, note.Note) or isinstance(el, chord.Chord) or isinstance(el, note.Rest):
					song.append(get_element_str(el, includeDuration))

			# Add the song to the list
			if byGenre:
				if genre in dataset:
					dataset[genre].append(song)
				else:
					dataset[genre] = [song]

			dataset["total"].append(song)

	with open(fileString, "wb") as output:
		pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)

	return dataset

def build_vocab(songs, includeDuration = False):
	vocab_set = set()
	for i in songs:
		vocab_set.update(i)

	vocab_dict = {}
	for index, element in enumerate(sorted(vocab_set)):
		vocab_dict[element] = index

	return vocab_dict

def standardize_songs(songs):

	max_song_length = max(map(len, songs))

	adjusted_songs = []
	for song in songs:

		song_length = len(song)
		full_addition = max_song_length // song_length
		part_addition = max_song_length % song_length

		adjusted_song = song*full_addition
		adjusted_song.extend(song[:part_addition])

		adjusted_songs.append(adjusted_song)

	return adjusted_songs

def rand_song_slice(song, slice_number, slice_length):
	start_element_i = math.floor(random.random() * (len(song) - slice_length - 1))
	end_element_i = start_element_i + slice_length + 1
	return song[start_element_i: end_element_i]

def slice_to_tensor(vocab, slice):
	out = torch.zeros(len(slice)).long()
	for i,j in enumerate(slice):
		out[i] = vocab[j]
	return out

def song_to_seq_target(vocab,song):
    a_slice = rand_song_slice(song,0,50)
    seq  = slice_to_tensor(vocab,a_slice[:-1])
    target = slice_to_tensor(vocab,a_slice[1:])
    assert(len(seq) == len(target)), 'SEQ AND TARGET MISMATCH'
    return Variable(seq), Variable(target)


class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model='lstm', num_layers=1):
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

def some_pass(loss_function,model,seq, target, fit=True):
    model.init_hidden() # Zero out the hidden layer
    model.zero_grad()   # Zero out the gradient
    some_loss = 0

    for i, c in enumerate(seq):
        output = model(c)
        print(output)
        print(target[i])
        some_loss += loss_function(output, target[i])
        
    if fit:
        some_loss.backward()
        optimizer.step()
    
    return some_loss.data[0] / len(seq)

def Train(vocab,songs):
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
        in_size, out_size = len(songs), len(songs)
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
    for epoch in range(start_epoch, N_EPOCHS):
        # Training
        for i, song in enumerate(songs):
            this_loss = some_pass(loss_function,model,*song_to_seq_target(vocab,song))
            loss += this_loss
            
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
                epoch, (i+1)/len(songs)*100, i, this_loss)
            sys.stdout.write(msg)
            sys.stdout.flush()
        print()
        losses.append(loss / len(songs))
            
        # Validation
        # for i, song_idx in enumerate(valid_idxs):
        #     this_loss = some_pass(*song_to_seq_target(data[song_idx]), fit=False)
        #     v_loss += this_loss
            
        #     msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
        #         epoch, (i+1)/len(valid_idxs)*100, i, toc(time_since), this_loss)
        #     sys.stdout.write(msg)
        #     sys.stdout.flush()
        # print()
        # v_losses.append(v_loss / len(valid_idxs))
        
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

if __name__ == '__main__':
	songs = get_dataset()
	vocab = build_vocab(songs['total'])
	l = standardize_songs(songs["total"])
	Train(vocab,l)
	
