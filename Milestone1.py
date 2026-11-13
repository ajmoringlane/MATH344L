# Written by Anthony Joseph Moringlane
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile

# Part A - small ratings matrix -AM

movies = ["Action 1", "Action 2", "Action 3",
          "Romance 1", "Romance 2", "Romance 3"]

users = ["Ana", "Ben", "Cal", "Dee", "Eve", "Fay"]

toy = np.array([
    [5, 4, 5, 1, np.nan, 2],
    [4, np.nan, 5, 2, 1, 1],
    [5, 5, np.nan, 1, 2, np.nan],
    [1, 2, 1, 5, 4, np.nan],
    [np.nan, 1, 2, 4, 5, 5],
    [2, 1, np.nan, 5, np.nan, 4]
])

toy_table = pd.DataFrame(toy, index=users, columns=movies)
print(toy_table)

missing = np.isnan(toy).sum()
print("Missing entries:", missing, "out of", toy.size)


# Part B - movielens data -AM

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

print("Ratings:", len(ratings))
print("Users:", ratings["user"].nunique())
print("Movies:", ratings["item"].nunique())

star_counts = ratings["rating"].value_counts().sort_index()
print(star_counts)

average = ratings["rating"].mean()
print("Average rating:", round(average, 3))

star_counts.plot(kind="bar", color="steelblue")
plt.xlabel("Stars")
plt.ylabel("Number of ratings")
plt.title("MovieLens 100K Ratings")
plt.show()


# Exercise 1 - guessing missing ratings -AM

predictions = {
    "Ana - Romance 2": 1,
    "Ben - Action 2": 5,
    "Cal - Action 3": 5,
    "Cal - Romance 3": 1,
    "Dee - Romance 3": 5,
    "Eve - Action 1": 1,
    "Fay - Action 3": 1,
    "Fay - Romance 2": 5
}

print(pd.Series(predictions))

print("Ben would give Action 2 a 5 since Ana and Cal liked it.")
print("Dee would give Romance 3 a 5 since Eve liked it.")


# Exercise 2 - adding another user -AM

new_user = [5, 4, np.nan, 1, 2, np.nan]

toy = np.vstack([toy, new_user])
users.append("Gus")

print(pd.DataFrame(toy, index=users, columns=movies))

print("Gus likes action more than romance.")
print("Action 3 prediction: 5")
print("Romance 3 prediction: 1")


# Exercise 3 - checking missing ratings -AM

print("Missing entries:", np.isnan(toy).sum())
print("Fraction missing:", np.isnan(toy).mean())

print("np.isnan(toy).sum() counts all missing ratings.")


# Exercise 4 - least common rating -AM

least_common = ratings["rating"].value_counts()

print("Least common rating:", least_common.idxmin())
print("Number of times:", least_common.min())


R = ratings.pivot_table(
    index="user",
    columns="item",
    values="rating"
)

print("Matrix shape:", R.shape)

fraction_missing = R.isna().to_numpy().mean()
print("Fraction missing:", round(fraction_missing, 4))

per_user = ratings.groupby("user").size()

print("Least active user:", per_user.min())
print("Median user:", int(per_user.median()))
print("Most active user:", per_user.max())

print("np.linalg.svd needs every entry to be a number, but R has NaN values.")
print("Filling in those missing entries is the recommendation problem itself.")


titles = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["item", "title"]
)

stats = ratings.groupby("item")["rating"].agg(["count", "mean"])
stats = stats.join(titles.set_index("item"))

top_five = stats.sort_values("count", ascending=False).head(5)
print(top_five.round(2))


print("Why a low-rank model is plausible:")
print("Ratings are driven by a few hidden preferences, like action vs romance")
print("or old vs new films. Each user has some amount of each preference and")
print("each movie has some amount of each feature. If only a few factors matter,")
print("the full ratings matrix can be approximated by a low-rank matrix.")


user_avg = ratings.groupby("user")["rating"].mean()
top_user = user_avg.idxmax()

print("Highest average user:", top_user)
print("Their average rating:", round(user_avg.max(), 3))
print("Number of ratings they gave:", per_user[top_user])
print("We should be careful calling them the most positive user because they")
print("may have rated only a few movies, and they may have only watched movies")
print("they expected to like. A small sample can give an extreme average.")


print("In ratings, one row is a single rating: a user, a movie, the stars, and")
print("a timestamp. It is long form with 100,000 rows.")
print("In R, one row is one user, with a column for every movie. Each entry is")
print("that user's rating of that movie, or NaN if they never rated it.")


print("With no missing entries, every value in the matrix is a number, so")
print("np.linalg.svd could run. But the ratings we want to predict are exactly")
print("the missing ones. A complete matrix would already contain the answers,")
print("and the SVD would just reproduce known ratings instead of predicting new")
print("ones. We would still need to pick a rank k and check that the low-rank")
print("version generalizes to ratings it has not seen.")


hidden_factors = pd.DataFrame({
    "High value for a user": [
        "Loves action and explosions",
        "Enjoys older classic films",
        "Prefers lighthearted comedies"
    ],
    "High value for a movie": [
        "Has lots of action scenes",
        "Was released decades ago",
        "Is funny and light in tone"
    ]
}, index=["Action vs romance", "Old vs new", "Comedy vs drama"])

print(hidden_factors)
