# This script will reformat the data so that instead of having words -> lines -> song
# we have just all the words in a line to apply the baseline LSTM
import json

newfile = 'top20flat/top20flattest.json'
open(newfile,'w').close()
with open('top20line/top20test.json','r') as f:
    for i, line in enumerate(f):
        line_dict = json.loads(line)
        genre = line_dict['genre']
        lyrics_idx = line_dict['idxs']
        flattened = [item for sublist in lyrics_idx for item in sublist]

        new_dict = {}
        new_dict['genre'] = genre
        new_dict['idxs'] = flattened

        with open(newfile,'a') as g:
            g.write(json.dumps(new_dict))
            g.write('\n')
        if (i % 50000 == 0):
            print("Completed:", i)
