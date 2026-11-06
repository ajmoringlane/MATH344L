# Written by Anthony Joseph Moringlane
#
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

# Part A - loading movielens data -AM

if not os.path.exists("ml-100k"):
    urllib.request.urlretrieve(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ml-100k.zip"
    )
    zipfile.ZipFile("ml-100k.zip").extractall()

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user", "item", "rating", "time"]
)

print(ratings.head())
print("Number of ratings:", len(ratings))


# Part B - splitting the ratings -AM

def split_ratings(ratings, test_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(ratings), dtype=bool)

    user_left = ratings.groupby("user").size().to_dict()
    item_left = ratings.groupby("item").size().to_dict()

    for pos in rng.permutation(len(ratings)):
        row = ratings.iloc[pos]
        user, item = row["user"], row["item"]

        if (rng.random() < test_frac
                and user_left[user] > 1
                and item_left[item] > 1):
            is_test[pos] = True
            user_left[user] -= 1
            item_left[item] -= 1

    return ratings.iloc[~is_test].copy(), ratings.iloc[is_test].copy()

train, test = split_ratings(ratings)

print("Train:", len(train))
print("Test:", len(test))
print("Test fraction:", round(len(test) / len(ratings), 3))


# Part C - making the training matrix -AM

N_USERS, N_ITEMS = 943, 1682

def to_matrix(df):
    R = np.full((N_USERS, N_ITEMS), np.nan)
    R[df["user"].to_numpy() - 1,
      df["item"].to_numpy() - 1] = df["rating"].to_numpy()
    return R

R = to_matrix(train)
known = ~np.isnan(R)

print("Matrix shape:", R.shape)
print("Known training ratings:", known.sum())

truth = test["rating"].to_numpy()
u = test["user"].to_numpy() - 1
i = test["item"].to_numpy() - 1


# Exercise 1 - small matrix and bias -AM

tiny = np.array([
    [5, 3, np.nan],
    [4, np.nan, 1],
    [np.nan, 2, 2]
])

mu_tiny = np.nanmean(tiny)
b_user = np.nanmean(tiny - mu_tiny, axis=1)

residual = tiny - mu_tiny - b_user[:, None]
b_item = np.nanmean(residual, axis=0)

print("Overall mean:", mu_tiny)
print("User bias:", b_user)
print("Movie bias:", b_item)


# Exercise 2 - checking what happens if we cheat -AM

all_R = to_matrix(ratings)
all_user_mean = np.nanmean(all_R, axis=1)

honest_user_mean = np.nanmean(R, axis=1)

def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2))

honest_score = rmse(honest_user_mean[u], truth)
cheat_score = rmse(all_user_mean[u], truth)

print("Honest RMSE:", round(honest_score, 4))
print("Cheating RMSE:", round(cheat_score, 4))
print("The cheating score uses test ratings, so it isnt a fair comparison.")
