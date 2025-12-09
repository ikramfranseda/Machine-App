import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model2 import load_and_preprocess, run_kmeans, pca_transform

st.set_page_config(page_title="Clustering Penyakit Kronis Anak", page_icon="📊", layout="wide")

st.title("📊 Clustering Penyakit Kronis Anak Usia Dini di Indonesia")

# Load Dataset
uploaded = st.file_uploader("📂 Upload Dataset CSV", type="csv")

DATASET_PATH = "chronic_disease_children_trend.csv"
if uploaded:
    df_raw_path = uploaded
else:
    df_raw_path = DATASET_PATH

df, X_scaled, feature_cols = load_and_preprocess(df_raw_path)
st.write("Dataset Loaded!")
st.dataframe(df.head())

# Sidebar, Pilih jumlah cluster
st.sidebar.header("🔧 Pengaturan Clustering")
K = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=3)

# Jalankan model K-Means
clusters, sil, dbi = run_kmeans(X_scaled, K)
df["Cluster"] = clusters

st.write(f"📌 Silhouette Score: **{sil:.4f}**")
st.write(f"📌 Davies-Bouldin Index: **{dbi:.4f}**")

# PCA Visualisasi
pca_result = pca_transform(X_scaled)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

st.subheader("🧩 Visualisasi Cluster")
st.scatter_chart(df, x="PC1", y="PC2", color="Cluster")

# Detail hasil cluster
st.subheader("📍 Hasil Kelompok Provinsi")
df_sorted = df.sort_values(by=["Province", "Year"])
st.dataframe(df_sorted[["Province", "Year"] + feature_cols + ["Cluster"]])


# Tren penyakit per cluster
if "Year" in df.columns:
    st.subheader("📈 Tren Penyakit per Cluster")
    for disease in feature_cols:
        fig, ax = plt.subplots()
        for cl in sorted(df["Cluster"].unique()):
            trend = df[df["Cluster"] == cl].groupby("Year")[disease].mean()
            ax.plot(trend.index, trend.values, marker='o', label=f"Cluster {cl}")
        ax.set_title(f"Tren {disease}")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Prevalensi (%)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# Download hasil
st.subheader("📥 Download Hasil Clustering")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="hasil_clustering.csv",
    mime="text/csv",
)
