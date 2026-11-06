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
