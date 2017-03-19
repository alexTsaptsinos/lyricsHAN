from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import json
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
import itertools

path2model = 'results/large/line/HANline.08-1.90.hdf5'
path2test = 'results/large/line/test.json'
MAX_WORDS = 10 # max number of words in line
#MAX_WORDS = 60 # for when by section
MAX_LINES = 60 # max number of lines in song
#MAX_LINES = 10 # for when by section
vocab_size = 30000 # and 1 for unknown, and 1 for mask
test_data_size = 49690 # CHANGE
attention_size = 100

# First load in test data
# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
labels = []
with open('flatdata/filteredgenrecounts.txt','r') as f:
    for i, line in enumerate(f):
        genre = ' '.join(line.split()[:-1])
        genre2idx[genre] = i
        idx2genre[i] = genre
        labels.append(genre)

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
test_labels = np.zeros((test_data_size,), dtype=int)
test_data = np.zeros((test_data_size, MAX_LINES, MAX_WORDS))
#test_labels = np.zeros((10,), dtype=int)
#test_data = np.zeros((10, MAX_LINES, MAX_WORDS))
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

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim = attention_size
        super(AttLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer = 'he_normal', trainable=True)
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True)
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = 'he_normal', trainable=True)
        self.trainable_weights = [self.W, self.bw, self.uw]
        super(AttLayer,self).build(input_shape)

    def call(self, x, mask=None):
        x_reshaped = tf.reshape(x, [K.shape(x)[0]*K.shape(x)[1], K.shape(x)[-1]])
        ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
        intermed = tf.reduce_sum(tf.multiply(self.uw, ui), axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x)[0], K.shape(x)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x*weights
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])



model = load_model(path2model, custom_objects={"AttLayer":AttLayer})
preds = model.predict(test_data)
preds_idxs = np.argmax(preds,axis=1)
#preds_idxs = [int(i) for i in preds_idxs]

print np.shape(preds)
print np.shape(test_labels)
print np.shape(preds_idxs)

print preds_idxs
print test_labels
#y_true = [int(i) for i in test_labels]

# fyi indexes
# rock is 71
# pop is 35
# alt is 3
# country is 33
# hip-hop/rap is 55
ar = confusion_matrix(test_labels, preds_idxs)
print ar

print labels
np.save('confusion_matrix.npy',ar)

plt.imshow(ar)
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)


thresh = ar.max() / 2.
for i, j in itertools.product(range(ar.shape[0]), range(ar.shape[1])):
    plt.text(j, i, ar[i, j],horizontalalignment="center", color="white" if ar[i, j] > thresh else "black")
