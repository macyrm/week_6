# %%
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%

# Load data, merge data, sklearn four steps (review documentation), lambda functions

# %%
# loading data, column titles in salary in first row
salary_data = pd.read_csv('2025_salaries.csv', header = 1, encoding = 'latin-1')

stats = pd.read_csv('nba_2025.txt', sep = ',', encoding = 'latin-1')

# %%
salary_data.head()
# issue here: Header row showing "unnamed," change header to 1. Salary data is also less than the stats data.

# %%
# help(pd.merge)
merged_data = pd.merge(salary_data, stats, on = 'Player')

# %%
# The row count of merged data dropped from 643 to 524 because duplicate values were removed. the merged data only includes rows where the player is mentioned in both datasets.

# %%
# Showing duplicates in the Player column
duplicates = merged_data[merged_data.duplicated(subset = 'Player', keep = False)]
print(duplicates)

# %%
# sklearn four steps:
# 1. create an instance of the model (ex: mymodel = KMeans(n_clusters = 3))
# 2. fit the model to the data (ex: mymodel.fit(X))
# 3. make predictions using the model (ex: predictions = mymodel.predict(X))
# 4. evaluate the model (ex: score = mymodel.score(X))

# for kmeans you do not need to predict and can just use the labels_ attribute
# to get the cluster assignments for each data point after fitting the model.

# pick the variable with the largest variance

# %%
# lambda functions are anonymous functions that can be defined in a single like of code
# they are often used for simple operations that can be expressed in a concise way
# for example, you can use a lambda function to create a new column in a
# dataframe that is the result of applying a simple operation to an existing column
# for example, you want to create a new column 'Salary_in_thousands' that
# is the salary divided by 1000, you can use a lambda function like
merged_data['Salary_in_thousands'] = merged_data['Salary'].apply(lambda x: x / 1000)
# the lambda syntax is: lambda arguments: expression
# in this case, the argument is 'x' which represents each value in the 'Salary'
# column, and the expression is 'x / 1000' which divides each salary by 1000. You can
# use operations and conditional statements in lambda functions. For example, if you want
# to create a new column 'High_Salary' that is True if the salary is greater than 1 million and False otherwise, you can use a lambda function
merged_data['High_Salary'] = merged_data['Salary'].apply(lambda x: True if x > 1000000 else False)
# for data manipulation and transformation in pandas.