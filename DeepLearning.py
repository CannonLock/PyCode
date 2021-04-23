import torch
import torch.nn.functional as F
#import torchtext
import time
import math
import random
import pandas as pd
import numpy as np
import glob
import pickle
from music21 import converter, instrument, note, chord

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
	end_element_i = start + slice_length
	return song[start_element, end_element_i]

def slice_to_tensor(vocab, slice):
	out = torch.zeros(len(seq)).long()


if __name__ == '__main__':
	songs = get_dataset()
	vocab = build_vocab(songs['total'])
	l = standardize_songs(songs["total"])
	print("fart")
