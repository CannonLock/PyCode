import torch
import torch.nn.functional as F
#import torchtext
import time
import random
import pandas as pd
import numpy as np
import glob
import pickle
from music21 import converter, instrument, note, chord

GENRES = ["Blues", "Country", "Indie", "Jazz", "Pop", "Psychedelic Rock", "Rock", "Soul"]

# TODO : (

def train_network():
	""" Train a Neural Network to generate music """
	notes = get_notes()

	# get amount of pitch names
	n_vocab = len(set(notes))

	network_input, network_output = prepare_sequences(notes, n_vocab)

	model = create_network(network_input, n_vocab)

	train(model, network_input, network_output)

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

def build_vocab(notes, includeDuration = False):
	vocab = set()
	for i in songs:
		vocab.update(i)

	return vocab

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


def prepare_sequences(notes):
	""" Prepare the sequences used by the Neural Network """
	sequence_length = 100
	vocab_length = len(set(notes))

	# get all pitch names
	pitchnames = sorted(set(notes))

	# create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / float(vocab_length)

	network_output = np_utils.to_categorical(network_output)

	return (network_input, network_output)

def create_network(network_input, n_vocab):
	""" create the structure of the neural network """
	model = Sequential()
	model.add(LSTM(
		512,
		input_shape=(network_input.shape[1], network_input.shape[2]),
		recurrent_dropout=0.3,
		return_sequences=True
	))
	model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
	model.add(LSTM(512))
	model.add(BatchNorm())
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(BatchNorm())
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	return model

def train(model, network_input, network_output):
	""" train the neural network """
	filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
	songs = get_dataset(regenerate=False)
	l = standardize_songs(songs["total"])
	print("fart")
