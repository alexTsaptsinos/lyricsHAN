# Here we link the lyrics data with the genre and artist
import json


# First let us load all the meta data into a dictionary. If multiple entries we just take the first one
global_dict = {} # this dict will have lyric id as key, and a subdict of artists, trackname, lyricid as value
with open('fullLight.json','r') as f:
    for i, line in enumerate(f):
        old_dict = json.loads(line)
        instrumental = bool(old_dict['instrumental'])
        if old_dict['lyricid'] in global_dict:
            continue
        if instrumental:
            continue
        if "artists" not in old_dict:
            continue
        if old_dict['lyricid'] == 0:
            continue
        artists = [x['artistname'].lower() for x in old_dict['artists']]

        new_dict = {}
        new_dict['artists'] = artists
        new_dict['trackname'] = old_dict['trackname'].lower()
        global_dict[old_dict['lyricid']] = new_dict
        if (i % 10000 == 0):
            print("Completed:", i)


print("Finished inputting lyrics to dict")
# Now we have a dict of all meta data, link with lyrics
with open('lyric.json','r') as f:
    for i, line in enumerate(f):
        # for each line in lyric, link with the relevant metadata
        if (i % 500000 == 0):
            print("At number:",i)
        dic = json.loads(line)

        lyricid = dic['lyricid']
        if lyricid == 0:
            continue
        if lyricid not in global_dict:
            continue

        new_dict = {}
        new_dict['artists'] = global_dict[lyricid]['artists']
        new_dict['trackname'] = global_dict[lyricid]['trackname']
        new_dict['lyrics'] = dic['lyrics']

        with open('fullLinked.json', 'a') as g:
            g.write(json.dumps(new_dict))
            g.write('\n')


