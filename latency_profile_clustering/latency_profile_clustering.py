# This script uses K-Means clustering to cluster traces in no-interference.csv file,
# based on their latency profiles
# and then analyzes the clusters to infer which paths they might correspond to.

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.manifold import TSNE

# Load your data
df = pd.read_csv('../../tracing-data.tar/tracing-data/social-network/no-interference.csv')

# Normalize latency data
scaler = MinMaxScaler()
feature_cols = df.columns[1:]  # Assuming the first column is 'trace_id' and rest are latencies
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Choose the number of clusters to correspond to the number of known execution paths
execution_paths = [
    [1, 2, 6, 14], [1, 2, 6, 15], [1, 3, 7, 23], [1, 3, 7, 24],
    [1, 4, 8, 12, 20], [1, 4, 9, 13, 29], [1, 4, 8, 12, 21], [1, 4, 9, 13, 30],
    [1, 4, 10, 17], [1, 4, 10, 18], [1, 5, 11, 26], [1, 5, 11, 27],
    [1, 2, 6, 16], [1, 3, 7, 25], [1, 4, 8, 12, 22], [1, 4, 9, 13, 31],
    [1, 4, 10, 19], [1, 5, 11, 28]
]
n_clusters = len(execution_paths)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit the model
X = df[feature_cols]
kmeans.fit(X)

# Predict the cluster for each trace
df['cluster'] = kmeans.predict(X)

with open('latency_profile_clustering.csv', 'w') as file:
    for cluster in range(n_clusters):
        # Get the average latencies for the cluster
        avg_latencies = df[df['cluster'] == cluster][feature_cols].mean().sort_values(ascending=False)

        # Write the cluster number as a header
        file.write(f"Cluster {cluster} Average Latencies:\n")

        # Write the sorted latencies to the file
        avg_latencies.to_csv(file, header=True)

# Analyze and print sorted latencies for each cluster
for cluster in range(n_clusters):
    print(f"\nCluster {cluster} Average Latencies:")
    # Get the average latencies for the cluster
    avg_latencies = df[df['cluster'] == cluster][feature_cols].mean().sort_values(ascending=False)
    print(avg_latencies)

# Now you can analyze the clusters to infer which paths they might correspond to
# For example, you might look at the average latency profile of each cluster
cluster_centers = kmeans.cluster_centers_
# Split each label and take the first part (the number before the first underscore)
short_feature_cols = [col.split('_')[0] for col in feature_cols]
# Plot the cluster centers to see what the average latency profile looks like for each cluster
plt.figure(figsize=(14, 7))
sns.heatmap(cluster_centers, annot=True, cmap='viridis', fmt=".2f", xticklabels=short_feature_cols)
plt.title('Average Latency Profile of Each Cluster')
plt.xlabel('Microservices')
plt.ylabel('Cluster Number')

# Save the plot as a PNG file
png_filename = 'no-interference'
plt.savefig(png_filename)

plt.show()

# # Apply t-SNE to reduce the dimensionality of the data
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)
#
# # Create a scatter plot with the t-SNE output
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['cluster'], palette='viridis', legend='full')
# plt.title('t-SNE Clustering of Traces')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.legend(title='Cluster')
# plt.show()
