# Import dependencies.
import pandas as pd
import gzip
from os import getcwd, listdir

# Courtesy to Julian McAuley, assistant professor at USCD.
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

# Courtesy to Julian McAuley, assistant professor at UCSD.
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient = 'index')
    
# To get a subset of the data: load the whole dataset first, then randomly sample the whole dataset, and get a subset
# of , for example, 100000 observations.

# Windows version.
# reviews_df = getDF("C:\\Users\\m5191\\Downloads\\reviews_Movies_and_TV.json.gz")
# meta_df = getDF("C:\\Users\\m5191\\Downloads\\meta_Moveis_and_TV.json.gz")

# Mac version.
meta_df = getDF("/Users/ray/Desktop/meta_Movies_and_TV.json.gz")
reviews_df = getDF("/Users/ray/Desktop/reviews_Movies_and_TV.json.gz")

# Randomly select 100000 observations.
random.seed(44)
random_idx = random.sample(range(1, len(reviews_df)+1), 100000)
reviews_subset = reviews_df.iloc[random_idx].reset_index()

# reviews_subset.to_csv(getcwd() + "/amazon_reviews.csv")
# reviews_df.to_csv(getcwd() + "/amazon_reviews_full.csv")
# meta_df.to_csv(getcwd() + "/amazon_meta.csv")
