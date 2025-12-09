import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

def load_and_preprocess(dataset_path):
    df = pd.read_csv(dataset_path)

    # Pilih fitur numerik (kecuali Province & Year)
    feature_cols = [c for c in df.columns if c not in ['Province', 'Year']]
    df_features = df[feature_cols].fillna(df[feature_cols].mean())

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return df, X_scaled, feature_cols

def run_kmeans(X_scaled, K):
    model = KMeans(n_clusters=K, random_state=42)
    clusters = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)
    dbi = davies_bouldin_score(X_scaled, clusters)
    return clusters, sil, dbi

def pca_transform(X_scaled):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    return pca_result
