import pickle as pk
import pandas as pd

# %% paths to the pkl files storing the reliability scores
path_ls = ['../data/results/reliability/reddit/askhistorians/glove.pkl',
           '../data/results/reliability/reddit/askhistorians/sgns.pkl',
           '../data/results/reliability/wikitext-103/glove.pkl',
           '../data/results/reliability/wikitext-103/sgns.pkl']

# %%
inter_metric_df = pd.DataFrame()

# %%
for path in path_ls:
    with open(path, 'rb') as f:
        data = pk.load(f)

    data_long = pd.Series(data['inter-rater'][0]).to_frame('score').rename_axis('target_w').reset_index()

    if 'glove' in path:
        data_long['model'] = 'glove'
    else:
        data_long['model'] = 'sgns'

    if 'askhistorians' in path:
        data_long['corpus'] = 'askhistorians'

    else:
        data_long['corpus'] = 'wikitext'

    inter_metric_df = pd.concat([inter_metric_df, data_long])

# %%
inter_metric_df

# %%
path_ls = ['../data/results/property/reddit_ask_historians.csv',
           '../data/results/property/wikitext.csv']

feature_df = pd.DataFrame()

for path in path_ls:
    data = pd.read_csv(path).rename({'Unnamed: 0': 'target_w'}, axis=1)

    if 'historians' in path:
        data['corpus'] = 'askhistorians'

    else:
        data['corpus'] = 'wikitext'

    feature_df = pd.concat([feature_df, data])

# %%
feature_df

# %% left join the features with the reliability data frame
full_df = pd.merge(inter_metric_df, feature_df, how="left", on=["target_w", "corpus"])

# %% check out the rows with NAs
full_df[full_df.isna().any(axis=1)]

# %% there are 400194 rows in total with missing values
full_df[full_df.isna().any(axis=1)].shape[0]

# %% that's 0.721213 missing
full_df[full_df.isna().any(axis=1)].shape[0] / full_df.shape[0]

# %% there are 156651 unique target words without features
len(full_df[full_df.isna().any(axis=1)].target_w.unique())

# %% there are 187886 unique target words (with reliability scores) in total
len(full_df.target_w.unique())

# %% there are 31235 unique target words (with both reliability and features) in total
len(full_df.target_w.unique()) - len(full_df[full_df.isna().any(axis=1)].target_w.unique())

# %% there are 31236 unique target words (with features) in total
len(feature_df.target_w.unique())

# %% check how many missing columns there are per row
full_df.isna().sum(axis=1).unique()
# this tells us that a target word either has all or no features

# %% keep only the rows without missing values
complete_df = full_df.dropna()
complete_df.to_csv('complete_inter_metric.csv', index=False) #save under the same directory (reliability_bias/mlr)