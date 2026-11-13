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


X_fit = np.where(known_fit, R_fit - B_fit, 0.0)
U2, s2, Vt2 = np.linalg.svd(X_fit, full_matrices=False)


ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
val_err1, val_err2 = [], []

for k in ks:
    P1 = rank_k(U1, s1, Vt1, k)
    P2 = rank_k(U2, s2, Vt2, k) + B_fit
    val_err1.append(score_at(P1, val_u, val_i, val_truth))
    val_err2.append(score_at(P2, val_u, val_i, val_truth))

best_k1 = ks[int(np.argmin(val_err1))]
best_k2 = ks[int(np.argmin(val_err2))]
print("Best Model 1 k:", best_k1, "validation RMSE:", round(min(val_err1), 4))
print("Best Model 2 k:", best_k2, "validation RMSE:", round(min(val_err2), 4))

os.makedirs("figures", exist_ok=True)

plt.semilogx(ks, val_err1, "o-", label="fill with movie averages")
plt.semilogx(ks, val_err2, "s-", label="SVD of baseline deviations")
plt.axhline(baseline_val_rmse, color="gray", linestyle="--",
            label="baseline")
plt.xlabel("rank k (log scale)")
plt.ylabel("validation RMSE (stars)")
plt.title("Choose rank using validation data")
plt.grid()
plt.legend()
plt.savefig("figures/validation_rmse_vs_k.png", dpi=150)
plt.show()


ks_iter = [2, 5, 10, 20]
val_err3 = []

for k in ks_iter:
    P3 = iterative_svd_predict(R_fit, k, n_iter=10)
    val_err3.append(score_at(P3, val_u, val_i, val_truth))

best_k3 = ks_iter[int(np.argmin(val_err3))]
print("Best iterative rank:", best_k3)
print("Best validation RMSE:", round(min(val_err3), 4))

val_summary = pd.Series({
    "baseline": baseline_val_rmse,
    f"Model 1 (k={best_k1})": min(val_err1),
    f"Model 2 (k={best_k2})": min(val_err2),
    f"Model 3 (k={best_k3})": min(val_err3),
}, name="validation RMSE")
print(val_summary.round(4).sort_values())


R_train = to_matrix(train)
test_truth = test["rating"].to_numpy()
test_u = test["user"].to_numpy() - 1
test_i = test["item"].to_numpy() - 1

B_train = baseline_matrix(R_train)
test_baseline = score_at(B_train, test_u, test_i, test_truth)

if min(val_err3) < min(val_err2) and min(val_err3) < min(val_err1):
    test_prediction = iterative_svd_predict(R_train, best_k3, n_iter=10)
    chosen_model = "iterative SVD"
    chosen_k = best_k3
    chosen_val = min(val_err3)
    chosen_ks, chosen_errs = ks_iter, val_err3
elif min(val_err2) < min(val_err1):
    known_train = ~np.isnan(R_train)
    X_train = np.where(known_train, R_train - B_train, 0.0)
    U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    test_prediction = rank_k(U, s, Vt, best_k2) + B_train
    chosen_model = "baseline plus low-rank deviations"
    chosen_k = best_k2
    chosen_val = min(val_err2)
    chosen_ks, chosen_errs = ks, val_err2
else:
    known_train = ~np.isnan(R_train)
    mu_train = np.nanmean(R_train)
    n_item_train = known_train.sum(axis=0)
    means_train = np.full(R_train.shape[1], mu_train)
    has_train = n_item_train > 0
    means_train[has_train] = (
        np.nansum(R_train, axis=0)[has_train] / n_item_train[has_train]
    )
    F_train = np.where(known_train, R_train, means_train[None, :])
    U, s, Vt = np.linalg.svd(F_train, full_matrices=False)
    test_prediction = rank_k(U, s, Vt, best_k1)
    chosen_model = "SVD of movie-average-filled matrix"
    chosen_k = best_k1
    chosen_val = min(val_err1)
    chosen_ks, chosen_errs = ks, val_err1

test_rmse = score_at(test_prediction, test_u, test_i, test_truth)
print("Chosen model:", chosen_model, "with k =", chosen_k)
print("Baseline test RMSE:", round(test_baseline, 4))
print("Chosen model test RMSE:", round(test_rmse, 4))


item_info = pd.read_csv(
    "ml-100k/u.item", sep="|", header=None, usecols=[0, 1],
    encoding="latin-1"
)
item_info.columns = ["item", "title"]

user_id = 1
user_row = user_id - 1
unrated = np.where(np.isnan(R_train[user_row]))[0]
predicted_scores = np.clip(test_prediction[user_row], 1, 5)
top = unrated[np.argsort(predicted_scores[unrated])[-5:][::-1]]

for col in top:
    title = item_info.loc[item_info["item"] == col + 1, "title"].iloc[0]
    print(title, "predicted rating:", round(predicted_scores[col], 2))


overfit_table = pd.DataFrame({
    "Model 1": pd.Series(val_err1, index=ks),
    "Model 2": pd.Series(val_err2, index=ks),
    "Model 3": pd.Series(val_err3, index=ks_iter),
}).round(4)
overfit_table.index.name = "rank k"
print(overfit_table)
print("Chosen model validation RMSE:", round(chosen_val, 4))
print("Chosen model test RMSE:", round(test_rmse, 4))
print("Gap (test - validation):", round(test_rmse - chosen_val, 4))
print("For Models 1 and 2, validation RMSE drops as k grows up to a point, then")
print("rises again. For Model 3 it is lowest at small k and gets worse as k grows.")
print("That rise at large k is overfitting: extra singular vectors fit noise in")
print("the known ratings instead of real taste patterns.")
print("The test RMSE is close to the best validation RMSE (slightly lower, since")
print("the final model was refit on all of train), so the chosen rank generalizes.")
print("If we picked k by looking at test scores, the test set would become")
print("another validation set and the final score would be too optimistic.")


mu_cold, b_user_cold, b_item_cold = bias_baseline(R_train)
cold_scores = np.clip(mu_cold + b_item_cold, 1, 5)
cold_top = np.argsort(cold_scores)[-5:][::-1]

print("Predictions for a brand new user with no ratings:")
for col in cold_top:
    title = item_info.loc[item_info["item"] == col + 1, "title"].iloc[0]
    print(title, "predicted rating:", round(cold_scores[col], 2))

print("A new user has no row of ratings, so there is no user effect and no")
print("low-rank preferences to learn. The best the model can do is the overall")
print("mean plus each movie's effect, which gives everyone the same list.")
print("That list is mostly obscure movies with one or two high ratings, so their")
print("movie effect is based on very little data.")
print("Asking the user to rate a few movies when they sign up, or using info")
print("like age, location, or favorite genres, would help personalize it.")


print(f"The baseline test RMSE is {test_baseline:.4f}: predicting with only the")
print("overall average, a user effect, and a movie effect is off by about that")
print("many stars on average.")
print(f"The chosen model ({chosen_model}, k = {chosen_k}) has test RMSE")
print(f"{test_rmse:.4f}, which is {test_baseline - test_rmse:.4f} stars better.")
print("The low-rank part captures shared taste patterns that the averages miss.")
print("The example recommendations show the movies the model scores highest for")
print("user 1 among movies they have not rated. They do not prove user 1 would")
print("like them; RMSE measures accuracy over many hidden ratings, not for any")
print("single recommendation. Most of the top picks are obscure movies with")
print("only a few ratings, so their high scores rest on very little evidence.")
