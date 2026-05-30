# ============================================================
#  Customer Segmentation with K-Means — Beginner Level
# ============================================================
#
#  Install required libraries (run once in your terminal):
#    pip install pandas scikit-learn matplotlib seaborn
#
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── 1. CREATE SAMPLE CUSTOMER DATA ──────────────────────────
#
#  In a real project you'd load a CSV:
#    df = pd.read_csv("customers.csv")
#
#  Here we build a small dataset manually so you can run
#  this script right away without any external files.

data = {
    "CustomerID": range(1, 21),
    "Annual_Spend":      [1200, 3500, 8000, 500,  9500, 4200, 700,
                          11000, 2800, 6500, 300, 7800, 4900, 1500,
                          8800, 600,  3100, 9200, 5500, 2200],
    "Visit_Frequency":   [2,    5,    9,    1,    12,   6,    1,
                          14,   4,    8,    1,    10,   7,    3,
                          11,   2,    5,    13,   7,    3],
}

df = pd.DataFrame(data)

print("=== Raw customer data (first 5 rows) ===")
print(df.head())
print(f"\nTotal customers: {len(df)}\n")


# ── 2. SELECT FEATURES FOR CLUSTERING ───────────────────────
#
#  We use two features:
#    • Annual_Spend     — how much a customer spends per year
#    • Visit_Frequency  — how often they visit per month

features = df[["Annual_Spend", "Visit_Frequency"]]


# ── 3. SCALE THE FEATURES ───────────────────────────────────
#
#  Annual_Spend is in the thousands; Visit_Frequency is 1–14.
#  Without scaling, K-Means will treat spend as far more
#  important just because its numbers are bigger.
#  StandardScaler brings both features to the same scale.

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("=== Scaled features (first 5 rows) ===")
print(pd.DataFrame(features_scaled,
                   columns=["Spend_scaled", "Frequency_scaled"]).head())
print()


# ── 4. FIND THE BEST K WITH THE ELBOW METHOD ────────────────
#
#  We train K-Means for K = 1 to 8 and record the inertia
#  (sum of squared distances from each point to its centroid).
#  The "elbow" — where the curve bends — is a good K choice.

inertias = []
k_range = range(1, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker="o", color="#378ADD", linewidth=2)
plt.title("Elbow Method — choosing the best K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster variance)")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150)
plt.show()
print("Elbow curve saved → elbow_curve.png\n")


# ── 5. TRAIN K-MEANS WITH K = 3 ─────────────────────────────
#
#  Based on the elbow curve, K=3 is usually a clear bend for
#  this dataset. Change K here if your curve suggests otherwise.

K = 3

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(features_scaled)

# Attach cluster labels back to the original dataframe
df["Cluster"] = kmeans.labels_

print("=== Customers with cluster labels ===")
print(df.to_string(index=False))
print()


# ── 6. DESCRIBE EACH CLUSTER ────────────────────────────────
#
#  Group by cluster and compute averages to understand
#  who lives in each segment.

summary = df.groupby("Cluster")[["Annual_Spend", "Visit_Frequency"]].mean()
summary.columns = ["Avg Annual Spend ($)", "Avg Monthly Visits"]
summary.index = [f"Cluster {i}" for i in summary.index]

print("=== Cluster summary ===")
print(summary.round(0))
print()


# ── 7. ASSIGN HUMAN-READABLE SEGMENT NAMES ──────────────────
#
#  After inspecting the summary, give each cluster a name.
#  These names are based on typical output — adjust to match
#  your actual cluster order (it can vary between runs).

def label_cluster(cluster_id, summary_df):
    """Return a segment name based on relative spend & frequency."""
    row = summary_df.loc[f"Cluster {cluster_id}"]
    spend = row["Avg Annual Spend ($)"]
    freq  = row["Avg Monthly Visits"]
    all_spends = summary_df["Avg Annual Spend ($)"]

    if spend == all_spends.max():
        return "High-Value VIPs"
    elif spend == all_spends.min():
        return "Occasional / At-Risk"
    else:
        return "Mid-Tier Regulars"

df["Segment"] = df["Cluster"].apply(
    lambda c: label_cluster(c, summary)
)

print("=== Customers with segment names (sample) ===")
print(df[["CustomerID", "Annual_Spend", "Visit_Frequency",
          "Cluster", "Segment"]].to_string(index=False))
print()


# ── 8. VISUALISE THE CLUSTERS ───────────────────────────────

palette = {0: "#378ADD", 1: "#D85A30", 2: "#1D9E75"}

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual_Spend",
    y="Visit_Frequency",
    hue="Cluster",
    palette=palette,
    s=100,
    edgecolor="white",
    linewidth=0.5,
)

# Plot centroids (un-scale them back to original units)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, (cx, cy) in enumerate(centroids_original):
    plt.scatter(cx, cy, marker="^", s=200,
                color=palette[i], edgecolor="white",
                linewidth=1.5, zorder=5,
                label=f"Centroid {i}" if i == 0 else "")

plt.title("Customer Segments (K-Means, K=3)")
plt.xlabel("Annual Spend ($)")
plt.ylabel("Monthly Visit Frequency")
plt.legend(title="Cluster", loc="upper left")
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=150)
plt.show()
print("Cluster plot saved → customer_clusters.png\n")


# ── 9. SAVE RESULTS TO CSV ──────────────────────────────────

df.to_csv("segmented_customers.csv", index=False)
print("Results saved → segmented_customers.csv")
print("\nDone! Your customers are split into these segments:")
print(df["Segment"].value_counts().to_string())
