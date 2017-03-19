# This script will create the training/dev/sets from the data
import random

# Clear the files
open('train.json','w').close()
open('dev.json','w').close()
open('test.json','w').close()

with open('indexed_data.json','r') as f:
    for line in f:
        r = random.uniform(0,1)
        if r < 0.8:
            with open('train.json','a') as g:
                g.write(line)
        elif r < 0.9:
            with open('dev.json','a') as g:
                g.write(line)
        else:
            with open('test.json','a') as g:
                g.write(line)


