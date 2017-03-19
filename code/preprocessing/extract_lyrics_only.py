# This script will go through all the lyrics and produce a text file of only the tokenized lyrics so that we can then perform analysis/get our vocab from the lyric corpus
import json
from nltk import sent_tokenize, word_tokenize
open('alllyrics.txt','w').close()

with open('lyric.json','r') as f:
    for i, line in enumerate(f):
        mydict = json.loads(line)
        lyrics = mydict['lyrics']
        lines = lyrics.split('\n')
        words = [word_tokenize(line) for line in lines]
        with open('alllyrics.txt','a') as g:
            for line in words:
                outstring = ' '.join(line)
                g.write(outstring)
                g.write('\n')

            g.write('\n')

        if (i % 50000 == 0):
            print("Completed:", i)
