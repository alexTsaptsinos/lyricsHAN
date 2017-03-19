# In this script we go through each lyrical datapoint and count the number of each genre seen
import json

genre_dict = {}
path = "all_data.json"
with open(path,"r") as f:
    for line in f:
        dic = json.loads(line)
        genre = dic['genre']
        if genre not in genre_dict:
            genre_dict[genre] = 1
        else:
            genre_dict[genre] += 1

# Now have counted all different genres, write to file
for key, val in genre_dict.items():
    with open("genrecounts.txt","a") as g:
        stringwrite = key + " " + str(val) + "\n"
        g.write(stringwrite)
