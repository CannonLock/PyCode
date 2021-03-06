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
import Model
import Training
import c
import predict

NUM_LAYERS, HIDDEN_SIZE = c.NUM_LAYERS, c.HIDDEN_SIZE
DROPOUT_P = c.DROPOUT_P

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

	fileString = "musicDataset_includeDuration=" + \
							 str(includeDuration) + \
							 "_byGenre=" + \
							 str(byGenre) + \
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

def build_translations(songs):
	vocab_set = set()
	for i in songs:
		vocab_set.update(i)

	str_to_int = {}
	int_to_str = []
	for index, element in enumerate(sorted(vocab_set)):
		str_to_int[element] = index
		int_to_str.append(element)

	return str_to_int, int_to_str

def data_composition(data):
	comp_dict = {}
	for song in data:
		for note in song:
			if note in comp_dict:
				comp_dict[note] += 1
			else:
				comp_dict[note] = 1

	return dict(sorted(comp_dict.items(), key=lambda item: item[1]))

def start_composition(data):
	start_dict = {}
	for song in data:
		if song[0] in start_dict:
			start_dict[song[0]] += 1
		else:
			start_dict[song[0]] = 1

	return dict(sorted(start_dict.items(), key=lambda item: item[1]))

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

if __name__ == '__main__':
	data = get_dataset()['total']
	standard_data = standardize_songs(data)
	str_to_int, int_to_str = build_translations(data)

	in_size, out_size = [len(str_to_int)]*2

	loss_function = nn.CrossEntropyLoss()
	model = Model.MusicRNN(in_size, HIDDEN_SIZE, out_size, NUM_LAYERS, DROPOUT_P)

	t = Training.ModelTrainer(loss_function, standard_data, str_to_int, model)
	t.final_training()
	
	#model = Model.MusicRNN(in_size, HIDDEN_SIZE, out_size, NUM_LAYERS, DROPOUT_P)
	#model.load_state_dict(torch.load('checkpoint/ckpt_mdl_lstm_ep_50_hsize_32_dout_0.9.pt'))
	#model.eval()

	# Create a song
	for i in range(5):
		start_note = list(start_composition(data).keys())[-i]
		song = predict.generate_song(model, str_to_int[start_note], 500)
		song_notes = predict.ints_to_notes(song, int_to_str)
		print(song_notes)
		predict.create_midi(song_notes, "test_output" + str(i) + ".mid")
 