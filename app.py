"""Streamlit app: predict the customer segment. Run: streamlit run app.py"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")
st.title("🛍️ Customer Segmentation (Hierarchical Clustering)")

path = "outputs/customers_with_clusters.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
else:
    st.error("Run hierarchical_customer_clustering.py first to generate the data.")
    st.stop()

features = ["Annual Income (k$)", "Spending Score (1-100)"]
scaler = StandardScaler().fit(df[features])
X = scaler.transform(df[features])
# Hierarchical clustering cannot predict new points, so we assign new customers
# to the cluster of their nearest neighbours.
knn = KNeighborsClassifier(n_neighbors=5).fit(X, df["Cluster"])

income = st.slider("Annual Income (k$)", 10, 140, 60)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Segment"):
    cluster = int(knn.predict(scaler.transform([[income, score]]))[0])
    st.success(f"Predicted Cluster: {cluster}")
    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[features[1]], c=df["Cluster"], cmap="Set1", alpha=0.6)
    ax.scatter(income, score, c="black", s=250, marker="*", label="New customer")
    ax.set_xlabel(features[0]); ax.set_ylabel(features[1]); ax.legend()
    st.pyplot(fig)

st.subheader("Cluster summary")
st.dataframe(pd.read_csv("outputs/cluster_profile.csv"))
