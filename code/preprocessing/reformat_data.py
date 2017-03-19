# This script will reformat the data so that instead of having words -> lines -> song
# we have words -> segments -> songs
# We guess a segment of a song when a word vector is blank
import json

newfile = 'top20section/top20sectiontest.json'
open(newfile,'w').close()
with open('top20line/top20test.json','r') as f:
    for i, line in enumerate(f):
        line_dict = json.loads(line)
        genre = line_dict['genre']
        lyrics_idx = line_dict['idxs']
        new_lyrics = []
        cur_seg = []
        for l in lyrics_idx:
            if len(l) == 0:
                new_lyrics.append(cur_seg)
                cur_seg = []
            else:
                cur_seg.extend(l)

        new_dict = {}
        new_dict['genre'] = genre
        new_dict['idxs'] = new_lyrics

        with open(newfile,'a') as g:
            g.write(json.dumps(new_dict))
            g.write('\n')

        if (i % 50000 == 0):
            print("Completed:", i)
