# Import dependencies.
from __future__ import division
import pandas as pd
from os import getcwd, listdir, path
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
from collections import Counter
import findspark
findspark.init()
import pyspark
import time
from IPython.display import Image
from IPython.display import HTML
from IPython.display import display
from PIL import Image
import urllib
import requests
import json
import argparse
from pyspark.mllib.recommendation import ALS
import math

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return(ID_and_ratings_tuple[0], (nratings, float(sum(float(x) for x in ID_and_ratings_tuple[1]))/nratings))

def recommend_movie(options):

	start_time = time.time()

	# Input all the data and combine them into one dataframe.
	ratings = pd.read_csv(getcwd()+"/ml-latest/ratings.csv")
	movies = pd.read_csv(getcwd()+"/ml-latest/movies.csv")
	links = pd.read_csv(getcwd()+"/ml-latest/links.csv")
	result = ratings.merge(movies, on = "movieId")
	result = result.merge(links, on = "movieId")
	reviews_df = result.sort_values(by = "userId").reset_index(drop = True)

	# Set optimal parameters obtained by training the dataset.
	seed = 5L
	best_rank = 8
	regularization_parameter = 0.25
	iterations = 10

	sc = pyspark.SparkContext()

	full_data = sc.textFile(path.join(getcwd(), "ml-latest", "ratings.csv"))
	small_data_header = full_data.take(1)[0]

	# Parse the data and read in one line of form (user, movie, rating) each time. 
	small_ratings_data = (full_data.filter(lambda line: line != small_data_header)
	                      .map(lambda line: line.split(","))
	                      .map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache())

	complete_movies_raw_data = sc.textFile(path.join(getcwd(), "ml-latest", "movies.csv"))
	complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

	# Parse 
	complete_movies_data = (complete_movies_raw_data.filter(lambda line: line != complete_movies_raw_data_header)
																.map(lambda line: line.split(","))
																.map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache())

	complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

	movie_ID_with_ratings_RDD = small_ratings_data.map(lambda x: (x[1], x[2])).groupByKey().mapValues(list)
	movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
	movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (int(x[0]), x[1][0]))

	# Obtain all movie id's and a list of ratings
	movie_id = movies.movieId
	rating = range(1, 6)

	new_user_ID = options.u
	num_movie = options.num_movie
	num_rating = num_movie

	new_user_ratings = zip(list(np.repeat(new_user_ID, num_movie)), 
		list(np.random.choice(movie_id, num_movie)), list(np.random.choice(rating, num_rating)))

	new_user_ratings_RDD = sc.parallelize(new_user_ratings)
	complete_data_with_new_ratings_RDD = small_ratings_data.union(new_user_ratings_RDD)

	# Everytime a new user is added to the dataset, the model needs to be trained again.
	new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, 
									seed = seed, iterations = iterations, lambda_ = regularization_parameter)

	new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)

	new_user_unrated_movies_RDD = complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0]))

	# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() 
	# to predict new ratings for the movies that new user has not seen.
	new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

	new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
	new_user_recommendations_rating_title_and_count_RDD = (new_user_recommendations_rating_RDD.join(complete_movies_titles)
															.join(movie_rating_counts_RDD))

	new_user_recommendations_rating_title_and_count_RDD = (new_user_recommendations_rating_title_and_count_RDD
															.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1])))

	top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2] >= 25).takeOrdered(25, key = lambda x: -x[1])

	print "Starting recommending movies..."
	print "---------------------------------------------------------------------------------------"

	print new_user_ratings

	print ("Top recommended movies (with more than 25 reviews):\n%s" % "\n".join(map(str, top_movies)))
	print "---------------------------------------------------------------------------------------"
	print "{} seconds elapsed".format(time.time()-start_time)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description = "Movie Recommendation")
	parser.add_argument("--u", help = "User ID", default = 0, type = int)
	parser.add_argument("--num-movie", help = "Number of movies this user has watched", default = 5, type = int)

	args = parser.parse_args()
	recommend_movie(args)



