import lastfm
import os
import numpy as np
import numpy.ma as ma
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

data_dir = '/Users/anubhavrana/Downloads'

tags_file = os.path.join(data_dir, 'lastfm_tags.json')
artists_file = os.path.join(data_dir, 'lastfm_artists.json')
#user_artists_file = os.path.join(data_dir, 'lastfm_user_artists.json')
#user_taggedartists_file = os.path.join(data_dir, 'lastfm_user_taggedartists.json')
#user_taggedartists_timestamps_file = os.path.join(data_dir, 'lastfm_user_taggedartists-timestamps.json')
user_friends_file = os.path.join(data_dir, 'lastfm_friends.json')
applied_tags_file = os.path.join(data_dir, 'lastfm_applied_tags.json')
plays_file = os.path.join(data_dir, 'lastfm_plays.json')

tags = pd.read_json(tags_file, lines=True)
artists = pd.read_json(artists_file, lines=True)
#user_artists = pd.read_json(user_artists_file, lines=True)
#user_taggedartists = pd.read_json(user_taggedartists_file, lines=True)
#user_taggedartists_timestamps = pd.read_json(user_taggedartists_timestamps_file, lines=True)
user_friends = pd.read_json(user_friends_file, lines=True)
applied_tags = pd.read_json(applied_tags_file, lines=True)
plays = pd.read_json(plays_file, lines=True)

#artists and plays is combined
#tags, applied_tags and user_friends is combined
artist_plays = pd.merge(artists, plays)
tags_friends = pd.merge(pd.merge(tags, applied_tags), user_friends)
#artist_plays.name.value_counts().head() #most commonly played artists

#New data structure that has artist and total number of plays 
artist_stats = artist_plays.groupby('name').agg({'plays': 'sum'})

#Min plays to be a top artist
min_plays = 5000

#Most popular artists
top_artists = artist_stats['plays'] >= min_plays
#artist_stats[top_artists].sort_values(by=('plays'), ascending=False).head() #display most popular artists

#Make a pivot table containing plays indexed by user id and artist id
tmp_df = plays.pivot(index='user_id', columns='artist_id', values='plays')

#Grouping artist together, sorting by descending order and selecting the ones that have more plays that play_count
play_count = 50
atrs = plays.groupby('artist_id').size().sort_values(ascending=False)
tmp_plays = plays.ix[atrs[atrs > play_count].index].dropna()

#New pivot table 
tmp_df = tmp_plays.pivot(index='user_id', columns='artist_id', values='plays')

#Reduced artist plays matrix --> if artist has more than 150 plays, value is 1, else, value is 0
art_plays_df = tmp_df.applymap(lambda x: 1 if x > 150 else 0).as_matrix()

#cosine similarity function
def cosine_similarity(u, v):
    return(np.dot(u, v)/np.sqrt((np.dot(u, u) * np.dot(v, v))))

# The user-artist matrix
x = art_plays_df

# Make a fake user 
y = np.zeros(the_data.shape[1], dtype=np.int32)
for i in np.where(x == 1)[1]:
    y[i] = 1
#y[2] = 1 ; y[11] = 1; y[15] = 1; y[78] = 1; y[136] = 1
#y[180] = 1; y[284] = 1; y[285] = 1; y[274] = 1; y[156] = 1

# Add a special index column to map the row in the x matrix to the user_ids
tmp_df.tmp_idx = np.array(range(x.shape[0]))

# Cosine similarity and maximum value
sims = np.apply_along_axis(cosine_similarity, 1, x, y)
mx = np.nanmax(sims)

# Finding the best matching user
usr_idx = np.where(sims==mx)[0][1]

# Printing the first thirty plays of test user and matched user.
print(y[:40])
print(x[usr_idx, :40])

print('\nCosine Similarity(y, x[{0:d}]) = {1:4.3f}' \
      .format(usr_idx, cosine_similarity(y, x[usr_idx])), end='\n\n')

# Now subtracting the vectors
# (any negative value is an artist to recommend)
art_vec = y - x[usr_idx]

# Making a mask array, so zeroing out any recommended artist.
art_vec[art_vec >= 0] = 1
art_vec[art_vec < 0] = 0

print(art_vec[:40])

# Printing out the number of artists we will recommend.
print('\n{0} Artist Recommendations for User = {1}' \
      .format(art_vec[art_vec == 0].shape[0], 
              tmp_df[tmp_df.tmp_idx == usr_idx].index[0]))

# Getting the columns (artist_ids) for the current user
art_ids = tmp_df[tmp_df.tmp_idx == usr_idx].columns

# Now making a masked array to find artists to recommend
# values are the artist ids, mask is the artists the most
# similar user liked.
ma_art_idx = ma.array(art_ids, mask = art_vec)
art_idx = ma_art_idx[~ma_art_idx.mask]

# Now making a DataFrame of the artists of interest and displaying
art_df = artists.ix[artists.artist_id.isin(art_idx)].dropna()

print(50*'=')

for artist in art_df.name.values:
    print(artist)

print(50*'=')

#Attempt at Multi-user recommendation

from sklearn.cross_validation import train_test_split

x, y = art_plays_df, range(art_plays_df.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=42)

# Adding an index into the user-artist DataFrame for the artists that are in the
# user-artist matrix.
tmp_df.tmp_idx = np.array(y)

# Iterating through each user in test set.
for idx, user in enumerate(x_test):
    
    # Cosine similarity, finding maximum value
    sims = np.apply_along_axis(cosine_similarity, 1, x_train, user)
    mx = np.nanmax(sims)
    
    # If maximum value is a real value    
    if mx > 0:
        
        # Finding the index in the similarity matrix with maximum value
        train_idx = np.where(sims==mx)[0][0]
        
        # Now subtracting the vectors 
        # (any negative value is an artist to recommend)
        art_vec = user - x_train[train_idx]
        
        # Making a mask aray, so zeroing out any recommended artist.
        art_vec[art_vec >= 0] = 1
        art_vec[art_vec < 0] = 0
        
        # y_train has the indices into the original
        # temporary data frame

        user_idx = tmp_df[tmp_df.tmp_idx == y_train[train_idx]]

        # State how many artists are being recommended for this user id
        print('{0} Artist Recommendations for User = {1}' \
              .format(art_vec[art_vec == 0].shape[0], \
                      tmp_df[tmp_df.tmp_idx == y_test[idx]].index[0]))
        
        print(50*'=')
        # Now making a masked array to find artists to recommend
        # values are the artist ids, mask is the artists the most
        # similar user liked.
        ma_art_idx = ma.array(user_idx.columns, mask = art_vec)
        art_idx = ma_art_idx[~ma_art_idx.mask]
        
        # Now making a DataFrame of the artists of interest and display
        art_df = artists.ix[artists.artist_id.isin(art_idx)].dropna()
        for artist in art_df.name.values:
            print(artist)
            
        print(50*'=')
'''
References--- Data used from Last.fm website, http://www.lastfm.com

   You may also cite HetRec'11 workshop as follows:

   @inproceedings{Cantador:RecSys2011,
      author = {Cantador, Iv\'{a}n and Brusilovsky, Peter and Kuflik, Tsvi},
      title = {2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)},
      booktitle = {Proceedings of the 5th ACM conference on Recommender systems},
      series = {RecSys 2011},
      year = {2011},
      location = {Chicago, IL, USA},
      publisher = {ACM},
      address = {New York, NY, USA},
      keywords = {information heterogeneity, information integration, recommender systems
And also INFO 490 course at University of Illinois at Urbana-Champaign
'''
