# Alex - you have your final model. You can now predict!
from keras.models import load_model
import numpy as np
import json
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

path2model = 'results/top20/stackedgrus/stackedGRUsbest.hdf5'
path2test = 'results/top20/line/top20test.json'
MAX_WORDS = 10 # max number of words in line
#MAX_WORDS = 60 # for when by section
MAX_LINES = 60 # max number of lines in song
#MAX_LINES = 10 # for when by section
vocab_size = 30000 # and 1 for unknown, and 1 for mask
test_data_size = 44938 # CHANGE
attention_size = 100

# First load in test data
# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
with open('results/top20/line/top20filteredgenrecounts.txt','r') as f:
    for i, line in enumerate(f):
        genre = ' '.join(line.split()[:-1])
        genre2idx[genre] = i
        idx2genre[i] = genre

num_genres = len(genre2idx)
n_classes = num_genres
print "Number of genres:", num_genres
print "Finished loading genres"

def pad_line(line):
    '''Pads/truncates a song line to have length MAX_WORDS'''
    size = min(MAX_WORDS, len(line))
    to_add = MAX_WORDS-size
    new_line = line[:size] + to_add*[vocab_size+1]
    return new_line


def format_data(d):
    '''Formats a datapoint to have correct length and numpy array format'''

    # Use this zero vector when padding lines.
    zero_line_vector = MAX_WORDS*[vocab_size+1]
    ret = [pad_line(line) for line in d if len(line) > 0]

    # Now pad song to have correct number of lines
    size = min(MAX_LINES, len(ret))
    to_add = MAX_LINES - size
    ret = ret[:size]
    ret.extend(to_add*[zero_line_vector])
    ret = np.array(ret)

    return ret


# Get test data
test_labels = np.zeros((test_data_size,))
test_data = np.zeros((test_data_size, MAX_LINES, MAX_WORDS))
with open(path2test,'r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        test_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = format_data(datapoint)
        test_data[i,:,:] = arr

print "Finished inputting test data"

model = load_model(path2model)
evals = model.evaluate(test_data, test_labels)
print "Test accuracy:", evals
