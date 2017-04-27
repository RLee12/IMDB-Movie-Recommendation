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

# From now on we work on this subset of the original dataset.
reviews_df = pd.read_csv("amazon_reviews.csv")

# Number of different reviewer names in the data.
len(set(reviews_df.reviewerName))

# Plot the user rating distribution.

N_star_categories = 5
colors = np.array(['#E50029', '#E94E04', '#EEC708', '#A5F30D', '#62F610'])
star_labels = np.array([star_label+1 for star_label in range(N_star_categories)])
star_category_dist_fig = plt.figure(figsize = (10, 8), dpi = 100)
bar_plot_indices = np.arange(N_star_categories)
star_category_absolute_frequencies = reviews_df.overall.value_counts(ascending = True)
star_category_relative_frequencies = (np.array(star_category_absolute_frequencies)/
                                      float(sum(star_category_absolute_frequencies)))
rects = (plt.bar(bar_plot_indices, star_category_relative_frequencies, width = 1, 
                 color = sns.color_palette("YlOrRd", 5), alpha = .7))

for (idx, rect) in enumerate(rects):
        plt.gca().text(rect.get_x() + rect.get_width()/2., 1.05*rect.get_height(), 
                       '%.3f'%(star_category_relative_frequencies[idx]), ha = 'center', va = 'bottom')

plt.xticks(bar_plot_indices + .5, star_labels)
plt.xlabel('Rating Category')
plt.ylabel('Relative Frequency')
plt.ylim([0, 1])
plt.title('User Rating Distribution for {} Amazon Movie and TV Reviews'.format(len(reviews_df)))

plt.show()




