import numpy as np
import json
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras import optimizers

## DEFINE PARAMETERS

vocab_size = 30000 # and 1 for unknown, and 1 for mask
learning_rate = .01 # tuned on dev set
training_epochs = 100
batch_size = 64
embed_size = 100

# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
with open('top20flat/top20filteredgenrecounts.txt','r') as f:
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

# Will get the average word vector for a song to use for baseline logistic regression
# Returns np.array of mean vector
def get_average(song):
    # Get rid of zero length lists
    #song = [line for line in song if len(line) > 0]
    if len(song) == 0:
        return np.zeros([1,100])
    # Calculate average
    av = np.mean(np.array([embedding_matrix[word] for word in song]), axis=0)
    return av

# Get data
train_labels = np.zeros([359137,])
train_data = np.zeros([359137,100])
with open('top20flat/top20flattrain.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        genre = my_dict['genre'].encode('utf-8')
        train_labels[i] = genre2idx[genre]
        datapoint = my_dict['idxs']
        # Get the average for the song
        av = get_average(datapoint)
        train_data[i,:] = av

# Get dev data
dev_labels = np.zeros([45383,])
dev_data = np.zeros([45383,100])
with open('top20flat/top20flatdev.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        genre = my_dict['genre'].encode('utf-8')
        dev_labels[i] = genre2idx[genre]
        datapoint = my_dict['idxs']
        # Get the average for the song
        av = get_average(datapoint)
        dev_data[i,:] = av

# Get test data
test_labels = np.zeros([44938,])
test_data = np.zeros([44938,100])
with open('top20flat/top20flattest.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        genre = my_dict['genre'].encode('utf-8')
        test_labels[i] = genre2idx[genre]
        datapoint = my_dict['idxs']
        # Get the average for the song
        av = get_average(datapoint)
        test_data[i,:] = av


model = Sequential()
model.add(Dense(num_genres, input_dim=100, activation='softmax'))

optimizer = optimizers.RMSprop(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print "model fitting - Baseline Logistic Regression"
print model.summary()
earlystopping = EarlyStopping(monitor='val_loss', patience=3)
checkpointer = ModelCheckpoint(filepath='results/large/lr/lrbest.hdf5',verbose=1,save_best_only=True)
hist = model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels),
          nb_epoch=training_epochs, batch_size=batch_size, callbacks=[checkpointer, earlystopping])
print hist.history
model.save('results/large/lr/lr.h5')

evals = model.evaluate(test_data, test_labels)
print "Test accuracy:", evals
