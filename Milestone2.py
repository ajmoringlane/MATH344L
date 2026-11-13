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


print("RMSE is in stars. An RMSE near 1 means predictions are about one star off.")


mu = np.nanmean(R)
user_mean = np.nanmean(R, axis=1)

n_item = known.sum(axis=0)
item_mean = np.full(N_ITEMS, mu)
has_rating = n_item > 0
item_mean[has_rating] = (
    np.nansum(R, axis=0)[has_rating] / n_item[has_rating]
)

print("Overall average:", round(mu, 3))

scores = {}
scores["always 3"] = rmse(np.full(len(test), 3.0), truth)
scores["overall average"] = rmse(np.full(len(test), mu), truth)
scores["user average"] = rmse(user_mean[u], truth)
scores["movie average"] = rmse(item_mean[i], truth)

for name, score in scores.items():
    print(f"{name:16s} {score:.4f}")


def bias_baseline(R):
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]

    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_rating = n_item > 0
    b_item[has_rating] = (
        np.nansum(residual, axis=0)[has_rating] / n_item[has_rating]
    )

    return mu, b_user, b_item

mu, b_user, b_item = bias_baseline(R)
pred = np.clip(mu + b_user[u] + b_item[i], 1, 5)

scores["user + movie effects"] = rmse(pred, truth)
print(pd.Series(scores, name="test RMSE").round(4).sort_values())


hand_mu = 17 / 6
hand_b_user = np.array([4 - hand_mu, 2.5 - hand_mu, 2 - hand_mu])
hand_b_item = np.array([(1 + 1.5) / 2, (-1 + 0) / 2, (-1.5 + 0) / 2])

print("By hand:")
print("mu =", round(hand_mu, 4))
print("b_user =", hand_b_user.round(4))
print("b_item =", hand_b_item.round(4))

check_mu, check_b_user, check_b_item = bias_baseline(tiny)

print("From bias_baseline:")
print("mu =", round(check_mu, 4))
print("b_user =", check_b_user.round(4))
print("b_item =", check_b_item.round(4))

print("Matches hand work:",
      np.isclose(check_mu, hand_mu)
      and np.allclose(check_b_user, hand_b_user)
      and np.allclose(check_b_item, hand_b_item))


def all_scores(seed):
    tr, te = split_ratings(ratings, seed=seed)
    R_s = to_matrix(tr)
    known_s = ~np.isnan(R_s)
    truth_s = te["rating"].to_numpy()
    u_s = te["user"].to_numpy() - 1
    i_s = te["item"].to_numpy() - 1

    mu_s = np.nanmean(R_s)
    user_mean_s = np.nanmean(R_s, axis=1)
    n_item_s = known_s.sum(axis=0)
    item_mean_s = np.full(N_ITEMS, mu_s)
    has_s = n_item_s > 0
    item_mean_s[has_s] = np.nansum(R_s, axis=0)[has_s] / n_item_s[has_s]

    m, bu, bi = bias_baseline(R_s)

    return {
        "always 3": rmse(np.full(len(te), 3.0), truth_s),
        "overall average": rmse(np.full(len(te), mu_s), truth_s),
        "user average": rmse(user_mean_s[u_s], truth_s),
        "movie average": rmse(item_mean_s[i_s], truth_s),
        "user + movie effects": rmse(
            np.clip(m + bu[u_s] + bi[i_s], 1, 5), truth_s),
    }

seed_table = pd.DataFrame({
    "seed 0": pd.Series(scores),
    "seed 1": pd.Series(all_scores(1)),
})
seed_table["difference"] = seed_table["seed 1"] - seed_table["seed 0"]
print(seed_table.round(4).sort_values("seed 0"))

same_order = (list(seed_table["seed 0"].sort_values().index)
              == list(seed_table["seed 1"].sort_values().index))
print("Same ranking for both seeds:", same_order)
print("The RMSE values change a little with a different split, because a")
print("different set of ratings is hidden. The ranking of the models stays the")
print("same, so the gaps between models are bigger than the split noise. When")
print("comparing two recommenders, use the same split for both, and if their")
print("scores are very close, check several seeds before claiming one is better.")
