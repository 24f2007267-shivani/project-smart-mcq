# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wandb

# %%
# Loading data
train = pd.read_csv(r"C:\Users\Shivani\OneDrive\Desktop\BS Degree\DL-GenAI-Project\data\train.csv")
test = pd.read_csv(r"C:\Users\Shivani\OneDrive\Desktop\BS Degree\DL-GenAI-Project\data\test.csv")

print(train.head())
print(train.shape, test.shape)

#%%
tr, val = train_test_split(
    train,
    test_size = 0.2,
    random_state=42, 
    stratify=train["answer"]
)

# %%
# MAP@3 scoring function
def map_at_3(true_labels, pred_rankings):
    scores = []
    for true, preds in zip(true_labels, pred_rankings):
        preds = preds[:3]
        if true not in preds:
            scores.append(0)
        else:
            rank = preds.index(true) + 1
            scores.append(1 / rank)
    return np.mean(scores)

# %%
# TF-IDF baseline model
options = ["A", "B", "C", "D", "E"]

def tfidf_rank(val_df):
    preds = []
    for _, row in val_df.iterrows():
        texts = [row["prompt"]] + [row[opt] for opt in options]
        vec = TfidfVectorizer().fit_transform(texts)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        ranked = [options[i] for i in np.argsort(-sims)]
        preds.append(ranked)
    return preds

# %%
# Validate
val_preds = tfidf_rank(val)
score = map_at_3(val["answer"].tolist(), val_preds)
print("TF-IDF Validation MAP@3:", score)

#%%
wandb.init(entity="24f2007267-shivani-iit-madras",
    # Set the wandb project where this run will be logged.
    project="24f2007267-t22026",
    name = "login-test"
    )
wandb.log({"test": 1})
wandb.finish()
