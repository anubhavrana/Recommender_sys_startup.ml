# Recommender_sys_startup.ml

This is the recommender system challenge problem for the fellowship at startup.ml using the last.fm data set found at https://github.com/Lab41/hermes/wiki/Datasets

I have attempted at a single user recommendation and a multi user recommendation system, unfortunately the single user recommender system does not recommend any artists for the fake user I created but the multi user system does generate artist recommendations for different/multiple users. 

Few notes on how the system could be improved: 
  If the minimum number of plays per artist were to be increased or decreased that would affect the number of users that get matched. Increasing the plays upto a certain point should improve the accuracy of the recommendations while decreasing would create recommendations that not necessarily match. 
  Another aspect would be the number of users being matched. Currently the system works on matching a single user and basing the recommendations on that, if this was extended to more users (for example 5 or more), the system would be able to produce more accurate recommendations. This may result in fewer recommendations but would produce more accurate results. 
