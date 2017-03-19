# This script will go through every review and tokenize the text
import json
from nltk import sent_tokenize, word_tokenize

outfile = 'tokenized_data.json' # json file to contain every datapoint with text split into sentences
all_outfile = 'allreviews_tokenized.txt' # text file containing every sentence to train word vecs on

open(outfile,'w').close()
open(all_outfile,'w').close()

with open('thinned_data.json','r') as f:
    for i, l in enumerate(f):
        mydict = json.loads(l)
        text = mydict['text']
        sents = sent_tokenize(text)
        words = [word_tokenize(sent) for sent in sents]
        # write data to file containing just text
        with open(all_outfile,'a') as g:
            for sent in words:
                outstring = ' '.join(sent)
                g.write(outstring)
                g.write('\n')
            g.write('\n')

        # now create new dict for tokenized datapoint
        newdict = {}
        newdict['stars'] = mydict['stars']
        newdict['text'] = words
        with open(outfile,'a') as h:
            h.write(json.dumps(newdict))
            h.write('\n')

        if (i % 50000 == 0):
            print("Completed:", i)

