# customer_segmentation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load the Mall Customers dataset."""
    url = "https://raw.githubusercontent.com/kennedykwangari/Mall-Customer-Segmentation-Data/refs/heads/master/Mall_Customers.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    """Preprocess the data for clustering."""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def find_optimal_clusters(X_scaled):
    """Find the optimal number of clusters using the Elbow Method."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow_method.png')  # Save the plot
    plt.show()

def apply_kmeans(X_scaled, n_clusters=5):
    """Apply K-Means clustering and return the cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    return kmeans.fit_predict(X_scaled)

def visualize_clusters(df, cluster_labels):
    """Visualize the clusters."""
    df['Cluster'] = cluster_labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis')
    plt.title('Customer Segmentation')
    plt.savefig('clusters.png')  # Save the plot
    plt.show()

def main():
    """Main function to run the customer segmentation pipeline."""
    # Load data
    df = load_data()
    
    # Preprocess data
    X_scaled = preprocess_data(df)
    
    # Find optimal clusters
    find_optimal_clusters(X_scaled)
    
    # Apply K-Means clustering
    cluster_labels = apply_kmeans(X_scaled, n_clusters=5)
    
    # Visualize clusters
    visualize_clusters(df, cluster_labels)

if __name__ == "__main__":
    main()