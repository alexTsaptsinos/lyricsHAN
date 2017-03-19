import json
import time
import requests

# Get what line have processed until
with open('last_line.txt', 'r') as g:
    start_from = int(g.readline())

# Now get first line from the fullLight file
path = "fullLinked.json"
mydict = {}
ctr = 0
outfile = "linkedGenre.json"
with open(path, "r") as f:
    for i, l in enumerate(f):
        if i < start_from:
            continue
        elif i == 50000:
            break
        tempdic = json.loads(l)

        # we have pre-processed and linked so know we have artists and not instrumental
        artists = tempdic['artists']
        trackname = tempdic['trackname']

        # query the API
        apiquery = "http://itunes.apple.com/search?term="+trackname+"&media=music&entity=musicTrack&attribute=songTerm&limit=100"
        resp = requests.get(apiquery)
        if resp.status_code == 404:
            continue
        while resp.status_code != 200:
            # Something went wrong
            print('GET /tasks/ {}'.format(resp.status_code))
            time.sleep(120)
            resp = requests.get(apiquery)

        results = resp.json()['results']
        #print("Number of results from itunes:", len(results))
        for res in results:
            testArtist = res['artistName'].lower()
            if (testArtist in artists) and (res['trackName'].lower() == trackname):
                # trackname and artist match :)
                newdict = {}
                newdict['genre'] = res['primaryGenreName']
                newdict['lyrics'] = tempdic['lyrics']
                with open(outfile, 'a') as g:
                    g.write(json.dumps(newdict))
                    g.write('\n')
                break

        ctr+=1
        if ctr % 50 == 0:
            print("Completed", ctr)

        # Update last line read
        with open('last_line.txt', 'w') as h:
            h.write(str(i+1))

        # sleep to stop rate limits kicking in
        time.sleep(2.5)



