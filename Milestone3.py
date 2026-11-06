# Written by Anthony Joseph Moringlane
#
import os
import subprocess
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part A - loading our github stuff -AM

USER = "your-github-username"
REPO = "your-repository-name"

if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
else:
    subprocess.run(
        ["git", "clone", f"https://github.com/{USER}/{REPO}.git", REPO],
        check=True
    )

sys.path.insert(0, f"{REPO}/src")

from evaluate import split_ratings, to_matrix, rmse
from models import bias_baseline

# loading the ratings again -AM

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

train, test = split_ratings(ratings, seed=0)
fit, val = split_ratings(train, seed=1)

print("Fit:", len(fit))
print("Validation:", len(val))
print("Test:", len(test))


# Part B - setting up validation -AM

R_fit = to_matrix(fit)
known_fit = ~np.isnan(R_fit)

val_truth = val["rating"].to_numpy()
val_u = val["user"].to_numpy() - 1
val_i = val["item"].to_numpy() - 1

def score_at(P, rows, cols, truth):
    pred = np.clip(P[rows, cols], 1, 5)
    return rmse(pred, truth)


# Part C - making the baseline matrix -AM

def baseline_matrix(R):
    mu, b_user, b_item = bias_baseline(R)
    return mu + b_user[:, None] + b_item[None, :]

B_fit = baseline_matrix(R_fit)

baseline_val_rmse = score_at(
    B_fit, val_u, val_i, val_truth
)

print("Baseline validation RMSE:", round(baseline_val_rmse, 4))


# Part D - rank k function -AM

def rank_k(U, s, Vt, k):
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


# Part E - filling gaps with movie averages -AM

mu_fit = np.nanmean(R_fit)

n_item = known_fit.sum(axis=0)
item_mean = np.full(R_fit.shape[1], mu_fit)

has_rating = n_item > 0
item_mean[has_rating] = (
    np.nansum(R_fit, axis=0)[has_rating] / n_item[has_rating]
)

F = np.where(known_fit, R_fit, item_mean[None, :])

U1, s1, Vt1 = np.linalg.svd(F, full_matrices=False)


# Exercise 1 - trying different fill values -AM

ks = [2, 5, 10]

fills = {
    "Movie average": F,
    "Overall average": np.where(known_fit, R_fit, mu_fit),
    "3 stars": np.where(known_fit, R_fit, 3.0)
}

results = {}

for name, matrix in fills.items():
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    for k in ks:
        P = rank_k(U, s, Vt, k)
        results[(name, k)] = score_at(P, val_u, val_i, val_truth)

print(pd.Series(results).round(4))

best = min(results, key=results.get)
print("Best fill and rank:", best)
print("Best RMSE:", round(results[best], 4))

print("Filling with 0 is bad because 0 isnt a real star rating.")


# Exercise 2 - checking full rank -AM

X_fit = np.where(known_fit, R_fit - B_fit, 0.0)

U2, s2, Vt2 = np.linalg.svd(X_fit, full_matrices=False)

full_rank = len(s2)
P_full = rank_k(U2, s2, Vt2, full_rank) + B_fit

print("Full rank:", full_rank)
print("Difference at gaps:",
      np.max(np.abs(P_full[~known_fit] - B_fit[~known_fit])))

print("Full rank rebuilds the filled matrix.")
print("Missing values were set to 0 deviation, so predictions stay at baseline.")
print("This doesnt mean it learned new preferences.")


# Exercise 3 - testing iterations -AM

def iterative_svd_predict(R, k, n_iter=10):
    known = ~np.isnan(R)
    B = baseline_matrix(R)
    X = np.where(known, R - B, 0.0)

    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)
        X = np.where(known, R - B, L)

    return L + B

for rounds in [1, 2, 5, 10]:
    P = iterative_svd_predict(R_fit, 5, n_iter=rounds)
    score = score_at(P, val_u, val_i, val_truth)
    print("Rounds:", rounds, "RMSE:", round(score, 4))

print("More iterations dont always mean better predictions.")
