import numpy as np
import json
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dropout, GlobalMaxPooling1D
from keras.layers import Dense
from keras import optimizers

## DEFINE PARAMETERS

MAX_WORDS = 600 # max number of words in song

vocab_size = 30000 # and 1 for unknown, and 1 for mask
learning_rate = .01
training_epochs = 20
batch_size = 64
embed_size = 100
dropout = 0.5
n_hidden = 50 # number of hidden states in LSTM
n_steps = MAX_WORDS

# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
with open('flatdata/filteredgenrecounts.txt','r') as f:
    for i, line in enumerate(f):
        genre = ' '.join(line.split()[:-1])
        genre2idx[genre] = i
        idx2genre[i] = genre

num_genres = len(genre2idx)
n_classes = num_genres
print "Number of genres:", num_genres
print "Finished loading genres"

# Load the embedding matrix
embedding_matrix = np.load('flatdata/embeddingMatrix.npy')
print "Finished loading embeddings"

def pad_line(line):
    '''Pads/truncates a song line to have length MAX_WORDS'''
    size = min(MAX_WORDS, len(line))
    to_add = MAX_WORDS-size
    new_line = line[:size] + to_add*[vocab_size+1]
    return new_line

# Get train data
train_labels = np.zeros((395722,))
train_data = np.zeros((395722, MAX_WORDS))
with open('flatdata/flattrain.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        train_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = pad_line(datapoint)
        train_data[i,:] = arr

print "Finished inputting training data"

# Get dev data
dev_labels = np.zeros((49776,))
dev_data = np.zeros((49776, MAX_WORDS))
with open('flatdata/flatdev.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        dev_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = pad_line(datapoint)
        dev_data[i,:] = arr

print "Finished inputting dev data"

# Get test data
test_labels = np.zeros((49690,))
test_data = np.zeros((49690, MAX_WORDS))
with open('flatdata/flattest.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        # Add label
        genre = my_dict['genre'].encode('utf-8')
        test_labels[i] = genre2idx[genre]
        # Add datapoint
        datapoint = my_dict['idxs']
        arr = pad_line(datapoint)
        test_data[i,:] = arr

print "Finished inputting test data"

print np.shape(train_data)

model = Sequential()
model.add(Embedding(vocab_size + 2, embed_size, weights=[embedding_matrix], input_length=MAX_WORDS, trainable=True))
model.add(LSTM(n_hidden, activation='sigmoid', return_sequences=True))
model.add(Dropout(dropout))
model.add(GlobalMaxPooling1D())
model.add(Dense(num_genres, activation='softmax'))

optimizer = optimizers.RMSprop(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print "model fitting - Baseline LSTM"
print model.summary()
earlystopping = EarlyStopping(monitor='val_loss', patience=3)
checkpointer = ModelCheckpoint(filepath='results/lstm/lstmbest.hdf5',verbose=1,save_best_only=True)
hist = model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels),
          nb_epoch=training_epochs, batch_size=batch_size, callbacks=[checkpointer, earlystopping])
print hist.history
model.save('results/lstm/lstm.h5')

evals = model.evaluate(test_data, test_labels)
print "Test accuracy:", evals
