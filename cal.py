import pandas as pd 
df=pd.read_excel('Task1and2/train.xlsx')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib 
def calculation(i):
    features = df.iloc[:, :-1]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    saved_cluster_centers = joblib.load('kmeans_weights.pkl')
    predicted_cluster = kmeans.fit_predict(scaled_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    # Create a scatter plot of the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predicted_cluster, cmap='viridis', alpha=0.6)
    # Highlight the selected point
    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], 
                c='red', label=f'Selected Row (Cluster {predicted_cluster[i]})', edgecolors='black')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title('Clusters Visualized in 2D')
    plt.legend()
    plt.savefig('static/sample_plot.png')
