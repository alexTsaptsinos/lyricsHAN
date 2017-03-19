# This script just takes the first 400,000 entries from the yelp dataset

open('thinned_data.json','w').close()

with open('yelp_academic_dataset_review.json','r') as f:
    for i, line in enumerate(f):
        if i == 400000:
            break
        else:
            with open('thinned_data.json','a') as g:
                g.write(line)
