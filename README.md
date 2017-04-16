# STA 160 Project Proposal

### Questions of Interest

We are interested in developing a movie recommendation engine. The recommendation
engine will be able to suggest similar movies to users based on users’ watched movie history.
The use of an efficient and accurate recommendation engine is crucial in today’s internet
industry, because it can vastly reduce user's time effort to look up for similar product/service,
and therefore potentially enrich user experience and increase sales for e-commerce. Since the
dataset contains rich information, we might use the dataset to do the following research if
we have enough time: 1). plot a network graph of the partnership among actors; 2). use movie
reviews to perform sentiment analysis; 3). use movie reviews to find out the trending words of
the movie.

### Skill Sets

Rui Li: Unix, Python, R, HTML, AWS, Machine Learning

Shunguan Mai: Unix, Python, R, HTML, AWS, Machine Learning, Economics

Shengjie Shi: Python, R, HTML, Machine Learning, Economics

### Targeted Audience

Three types of people will be interested in this movie recommendation engine:

1. Movie audiences: the movie recommendation engine can help them efficiently locate
similar movies based on their appetites.

2. Movie retailers/e-commerce vendors: sales revenue will increase since customers are
more likely exposed to more movies which they are interested to purchase.

3. Movie investors: the project can help movie investors to decide whether to invest to
similar movies based on reviews.

### Data Source

In order to avoid bias, we will combine data from different sources.

Direct Data:

1. IMDB movie database: containing movie name, actor/actress, producer, director,
release date, sales of box office, etc. 
<br>Source: http://eeyore.ucdavis.edu/stat141/Data/imdbpy.db</br>

2. Amazon user reviews for movies and TV series containing review text, user
review count, user rating, etc.
<br>Source: http://jmcauley.ucsd.edu/data/amazon/</br>

Auxiliary Data:

- For now no applicable data are found for our project purpose. We will continue to
look for supplementary data as the project progresses.

### Projected Schedule:

1. _Data wrangling_: clean up the movie data, combine the data from different sources, and
indicates what variables should be included in the final dataset. 
<br>_Obstacle_: since the datasets are from different sources, we might undergo the hardship of joining the dataset together. 
If the datasets are too large, we might need a server to do the work.
<br>_Members_: Rui Li, Shunguan Mai
<br>_Time_: 1 ~ 2 weeks (week 3 - week 4)</br>

2. _Develop a recommendation engine_: read research papers, watch online tutorials, and use
the movie dataset to develop a recommendation engine. Try at least two recommendation
algorithms and compare difference between them.
<br>_Obstacle_: we might need to adjust the dataset again since different algorithms use
different data inputs.
<br>_Members_: Shengjie Shi, Shunguan Mai
<br>_Time_: 1 ~ 2 weeks (week 5 - week 6)</br>

3. _Validation_: test the accuracy and efficiency of the recommendation engine
<br>_Obstacle_: it’s hard to quantify the accuracy of the recommendation engine since we do
not have a test dataset. The best we can do is to manually check if the recommended
outputs are similar to the inputs. However, we can test the efficiency of the engine by
checking its elapsed time.
<br>_Members_: Rui Li, Shengjie Shi
<br>_Time_: 1 ~ 2 weeks (week 7 - week 8)</br>

4. _Written report_: write the report, explain our methodology, which possibly includes
graphics, etc.
<br>_Members_: Shunguan Mai, Shengjie Shi
<br>_Time_: 1 week (week 9)</br>

5. _Buffer time_: Considering some unexpected situations, such as more time to read the
reference paper or more time to complete the data wrangling. If the projected schedule
works as expected, we can use this time to test our final product.
<br>_Time_: 1 week (week 10)</br>
