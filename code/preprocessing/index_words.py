# This script will create the embedding matrix and save it as a numpy matrix,
# go through the lyrics and convert them to their index in the embedding matrix
import json
import numpy as np

top = 30000 # however many words we'll take
# First create dict of words and numpy embedding matrix
embedding_matrix = np.zeros((top+2,100))
word_dict = {}
with open('myvectors.txt','r') as f:
    for i, line in enumerate(f):
        inputs = line.split()
        word = inputs[0]
        vec = np.array([float(x) for x in inputs[1:]])
        embedding_matrix[i,:] = vec
        word_dict[word] = i

# Save the embedding matrix
np.save('embeddingMatrix',embedding_matrix)
print("Finished embeddings")

def link_word(word):
    if word in word_dict:
        return word_dict[word]
    else:
        return word_dict['<unk>']

# Now go through dataset indexing all the words correctly.
# Save each song as a numpy array
outfile = 'indexed_data.json'
open(outfile,'w').close()
with open('tokenized_data.json','r') as f:
    for i, line in enumerate(f):
        datapoint = json.loads(line)
        my_lines = datapoint['lyrics']
        idx_lines = []
        for l in my_lines:
            idxs = [link_word(word) for word in l]
            idx_lines.append(idxs)

        new_dict = {}
        new_dict['genre'] = datapoint['genre']
        new_dict['idxs'] = idx_lines
        with open(outfile,'a') as g:
            g.write(json.dumps(new_dict))
            g.write('\n')

        if (i % 50000 == 0):
            print("Completed:", i)
