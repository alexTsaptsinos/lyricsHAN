import json
# This script will go through all lyrics and remove any that are part of genres with less than 100 instances

min_number = 4000
# First load in all the genres
genre_dict = {}
outpath = 'top20indexed_data.json'
outpath2 = 'top20filteredgenrecounts.txt'
open(outpath2,'w').close()
with open('genrecounts.txt','r') as f:
    for line in f:
        split = line.split()
        genre = ' '.join(split[:-1])
        count = int(split[-1])
        if genre == "Music Videos":
            continue
        if count >= min_number:
            genre_dict[genre] = count
            with open(outpath2,'a') as g:
                string2write = genre + " " + str(count) + "\n"
                g.write(string2write)
print("Finished filtering genres")

# Now go through every datapoint filtering out those of too few genre
open(outpath,'w').close()
with open('indexed_data.json','r') as f:
    for i, line in enumerate(f):
        my_dict = json.loads(line)
        genre = my_dict['genre']
        if genre not in genre_dict:
            continue
        else:
            with open(outpath,'a') as g:
                g.write(json.dumps(my_dict))
                g.write('\n')
