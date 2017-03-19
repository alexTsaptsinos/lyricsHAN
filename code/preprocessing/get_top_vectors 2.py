# This script will get the top however many vectors from the embeddings to use

top = 30000
outfile = 'myvectors.txt'
open(outfile,'w').close()

with open('vectors100.txt','r') as f:
    for i, line in enumerate(f):
        if i < top:
            with open(outfile,'a') as g:
                g.write(line)

# get <unk> also
with open(outfile,'a') as g:
    g.write(line)
