# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
from ipywidgets import interact
import ipywidgets as widgets

# For Jupyter Notebook environments (optional, remove if running as a script)
# %matplotlib inline 

# Step 1: Load the Dataset
file_path = '/Users/jankipatel/Desktop/E-commerce.csv'  # Adjust the path as needed
data = pd.read_csv(file_path)

# Step 2: Explore the Dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nLast 5 rows of the dataset:")
print(data.tail())
print("\nMissing values in each column:")
print(data.isnull().sum())
print("\nBasic statistics of the dataset:")
print(data.describe())

# Step 3: Data Preprocessing
data = data.dropna()
data = data.rename(columns={'Days Since Last Purchase': 'Recency', 'Items Purchased': 'Frequency'})

# Step 4: Feature Selection and Scaling
features = data[['Total Spend', 'Frequency', 'Recency']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("\nScaled features:")
print(scaled_features)

# Step 5: Data Visualization
sns.pairplot(data[['Total Spend', 'Recency', 'Frequency']])
plt.show()
data[['Total Spend', 'Frequency', 'Recency']].hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()
sns.heatmap(data[['Total Spend', 'Frequency', 'Recency']].corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 6: Interactive Elbow Method
def plot_elbow(max_k):
    """Plot the Elbow Method graph to determine the optimal number of clusters."""
    inertia = []
    k_values = range(1, max_k + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

# Interactive slider for Elbow Method
interact(plot_elbow, max_k=widgets.IntSlider(value=8, min=1, max=15, step=1, description="Max K"))

# Step 7: Interactive KMeans Clustering Visualization
def plot_clusters(n_clusters):
    """Display a 3D scatter plot of clusters based on selected number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # 3D Scatter plot of the clusters
    fig = px.scatter_3d(
        x=scaled_features[:, 0], 
        y=scaled_features[:, 1], 
        z=scaled_features[:, 2],
        color=clusters,
        labels={'x': 'Total Spend', 'y': 'Frequency', 'z': 'Recency'},
        title=f'3D Scatter Plot with {n_clusters} Clusters'
    )
    fig.show()

# Interactive slider to adjust the number of clusters in 3D scatter plot
interact(plot_clusters, n_clusters=widgets.IntSlider(value=4, min=2, max=10, step=1, description="Clusters"))
