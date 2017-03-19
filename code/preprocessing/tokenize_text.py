# This script will go through every entry in the filtered data and reformat the lyrics into the desired tokenized form:
# [['I',''m','going','home'],[...],...,[...]]
from nltk import sent_tokenize, word_tokenize
import json

outfile = 'tokenized_data.json'

open(outfile,'w').close()

with open('filtered_data.json','r') as f:
    for i, l in enumerate(f):
        mydict = json.loads(l)
        lyrics = mydict['lyrics']
        lines = lyrics.split('\n')
        words = [word_tokenize(line) for line in lines]

        # now create new dict for tokenized datapoint
        newdict = {}
        newdict['genre'] = mydict['genre']
        newdict['lyrics'] = words
        with open(outfile,'a') as g:
            g.write(json.dumps(newdict))
            g.write('\n')

        if (i % 50000 == 0):
            print("Completed:", i)

