# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %% 
# help(pd.read_csv)
df = pd.read_csv('house_votes_Dem.csv', encoding = 'latin-1')

# %% 
df.head()

# %%
df.info()

# %%
# choosing numerical features
c_num = df[['aye', 'nay', 'other']]

# %%
# documentation for kmeans in sklearn
help(KMeans)

# %%
# parameters defined for fitting data, will train model off of these things
kmeans = KMeans(n_clusters = 3, random_state = 42, verbose = 1)
kmeans.fit(c_num)

# %%
# information from model
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# %%
# cluster labels for original df
df['cluster'] = kmeans.labels_

# %%
help(KMeans)

# %%
# for loop for different cluster numbers to 
# see how inertia is impacted
inertias = []
k_values = range(1,10)
for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(c_num)
    inertias.append(kmeans.inertia_)

# %%
# plot of inertia values to see if there is an elbow
plt.figure(figsize = (10, 5))
plt.plot(k_values, inertias, marker = 'o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# %%
