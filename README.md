# IMDB Movie Recommendation

## Abstract 

In this project we explored two different approaches to recommend movies to a potential user: collaborative filtering and content-based filtering. Our goal is to reproduce these two approaches and examine the performance of these approaches. 

It should be noticed that these two approaches are fundamentally different from each other. In short, they differ in the way of how to make recommendations. Collaborative filtering uses solely users' ratings towards certain movies in the past, based on the assumption that users who show similar taste in the past will show the same taste in the future. Content-based filtering takes more flexible forms, such as movie overview, user's review, movie poster, etc. 

For collaborative filtering, we explored and used Apache Spark. We concluded a RMSE 0.88, which is a very decent outcome. Regarding the content-based filtering (or movie poster in this context), it is more depending on movie posters themselves. Since the underlining algorithm of content-based filtering used in this project is designed to maximize the similarity of different movie posters, it is foreseeable that totally different movies will be recommended to a user who might not like those movies at all. This result is due to that the movie poster is at best a moderate indicator of the content of this movie. 
