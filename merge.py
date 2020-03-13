import pandas as pd
import glob

filenames = glob.glob("datasets/general tweets/*.json")
print(filenames)

dfs = list()
for filename in filenames:
    df = pd.read_json(filename, dtype={'text': str}, lines=True)
    dfs.append(df)

tweets = pd.concat(dfs)
tweets.to_json("Datasets/general_tweets.json", orient='records')


