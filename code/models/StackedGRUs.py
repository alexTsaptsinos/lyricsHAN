import tensorflow as tf
import numpy as np
import json
from keras.engine.topology import Layer
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import TimeDistributed, Bidirectional, Embedding, LSTM, GRU, Conv1D, Dropout
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling1D
from keras import backend as K
from keras import optimizers

## DEFINE PARAMETERS

MAX_WORDS = 10 # max number of words in line
MAX_LINES = 60 # max number of lines in song

vocab_size = 30000 # and 1 for unknown, and 1 for mask
learning_rate = .01
training_epochs = 10
batch_size = 64
embed_size = 100
dropout = 0.5
n_hidden = 50 # number of hidden states in one direction of bidirectional GRU/LSTM
attention_size = 100 # size of hidden layer output from word attention
#max_grad_norm = 1. # don't clip norm for this
n_steps = MAX_WORDS
n_input = MAX_LINES

# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
with open('linedata/filteredgenrecounts.txt','r') as f:
    for i, line in enumerate(f):
        genre = ' '.join(line.split()[:-1])
        genre2idx[genre] = i
        idx2genre[i] = genre

num_genres = len(genre2idx)
n_classes = num_genres
print "Number of genres:", num_genres
print "Finished loading genres"

# Load the embedding matrix
embedding_matrix = np.load('linedata/embeddingMatrix.npy')
print "Finished loading embeddings"

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


# Get train data
train_labels = np.zeros((395722,))
train_data = np.zeros((395722, MAX_LINES, MAX_WORDS))
with open('linedata/train.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        train_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = format_data(datapoint)
        train_data[i,:,:] = arr

print "Finished inputting training data"

# Get dev data
dev_labels = np.zeros((49776,))
dev_data = np.zeros((49776,MAX_LINES, MAX_WORDS))
with open('linedata/dev.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        dev_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = format_data(datapoint)
        dev_data[i,:,:] = arr

print "Finished inputting dev data"

print np.shape(train_data)
embedding_layer = Embedding(vocab_size + 2,
                            embed_size,
                            weights=[embedding_matrix],
                            input_length=MAX_WORDS,
                            trainable=True)

sentence_input = Input(shape=(MAX_WORDS,), dtype='int32', name='sentence_input')
embedded_sequences = embedding_layer(sentence_input)
l_gru = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init='he_normal', inner_activation='sigmoid'), name='bidirect_word')(embedded_sequences)
l_drop = Dropout(dropout)(l_gru)
l_pool = GlobalAveragePooling1D()(l_drop)
sentEncoder = Model(sentence_input, l_pool, name='sentence_encoder')

review_input = Input(shape=(MAX_LINES,MAX_WORDS), dtype='int32', name='review_input')
review_encoder = TimeDistributed(sentEncoder, name='review_encoder')(review_input)
l_gru_sent = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init = 'he_normal', inner_activation = 'sigmoid'), name='bidirect_sentence')(review_encoder)
l_drop_sent = Dropout(dropout)(l_gru_sent)
l_pool_sent = GlobalAveragePooling1D()(l_drop_sent)
preds = Dense(num_genres, activation='softmax')(l_pool_sent)
model = Model(review_input, preds)

optimizer = optimizers.RMSprop(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print "model fitting - Stacked GRUs"
print model.summary()
earlystopping = EarlyStopping(monitor='val_loss', patience=3)
checkpointer = ModelCheckpoint(filepath='results/stackedGRUs/stackedGRUsbest.hdf5',verbose=1,save_best_only=True)
hist = model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels),
          nb_epoch=training_epochs, batch_size=batch_size, callbacks=[checkpointer, earlystopping])
print hist.history
model.save('results/stackedGRUs/stackedGRUs.h5')
