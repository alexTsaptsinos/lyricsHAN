import tensorflow as tf
import numpy as np
import json
from keras.engine.topology import Layer
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed, Bidirectional, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Dropout
from keras.layers import Dense, Input
from keras import backend as K
from keras import optimizers

## DEFINE PARAMETERS

MAX_WORDS = 10 # max number of words in line
MAX_LINES = 60 # max number of lines in song

vocab_size = 30000 # and 1 for unknown, and 1 for mask
learning_rate = .01 # tuned on dev set
training_epochs = 10
batch_size = 64
embed_size = 100
dropout = 0.5 # tuned on dev set
n_hidden = 50 # number of hidden states in one direction of bidirectional GRU/LSTM
attention_size = 100 # size of hidden layer output from word attention
max_grad_norm = 1. # tuned on dev set
n_steps = MAX_WORDS
n_input = MAX_LINES

# First load genre dict
# Create a dict for genres
genre2idx = {}
idx2genre = {}
with open('data/filteredgenrecounts.txt','r') as f:
    for i, line in enumerate(f):
        genre = ' '.join(line.split()[:-1])
        genre2idx[genre] = i
        idx2genre[i] = genre

num_genres = len(genre2idx)
n_classes = num_genres
print "Number of genres:", num_genres
print "Finished loading genres"

# Load the embedding matrix
embedding_matrix = np.load('data/embeddingMatrix.npy')
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
with open('data/train.json','r') as f:
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
with open('data/dev.json','r') as f:
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


sentence_input = Input(shape=(MAX_WORDS,), dtype='int32', name='sentence_input')
embedded_sequences = embedding_layer(sentence_input)
l_gru = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init='he_normal', inner_activation='sigmoid'), name='bidirect_word')(embedded_sequences)
l_att = AttLayer()(l_gru)
l_drop = Dropout(dropout)(l_att)
sentEncoder = Model(sentence_input, l_drop, name='sentence_encoder')

review_input = Input(shape=(MAX_LINES,MAX_WORDS), dtype='int32', name='review_input')
review_encoder = TimeDistributed(sentEncoder, name='review_encoder')(review_input)
l_gru_sent = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init = 'he_normal', inner_activation = 'sigmoid'), name='bidirect_sentence')(review_encoder)
#l_dense_sent = TimeDistributed(Dense(200))(l_gru_sent)
l_att_sent = AttLayer()(l_gru_sent)
l_drop_sent = Dropout(dropout)(l_att_sent)
preds = Dense(num_genres, activation='softmax')(l_drop_sent)
model = Model(review_input, preds)

optimizer = optimizers.RMSprop(lr=learning_rate, clipnorm = max_grad_norm)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print "model fitting - Hierachical LSTM"
print model.summary()
checkpointer = ModelCheckpoint(filepath='results/bidirectGRUattention.{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1,save_best_only=True)
model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels),
          nb_epoch=training_epochs, batch_size=batch_size, callbacks=[checkpointer])
model.save('results/HANdrophalfgrad1.h5')
