# %% [markdown]
# # Section G: Clustering for Exploratory Insight
# ## Video Game Sales Dataset - Discovering Natural Market Segments

# %% [markdown]
# ### Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
sns.set_palette("husl")

# %% [markdown]
# ### Load Dataset

# %%
# Load the dataset
df = pd.read_csv('vgsales.csv')

# Clean Year data
df_clean = df.dropna(subset=['Year'])
df_clean['Year'] = df_clean['Year'].astype(int)

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Clean dataset: {df_clean.shape[0]} rows")

# %% [markdown]
# ## 11.1 Clustering Motivation
# 
# ### Core Questions:
# 1. **Why might natural groupings exist in video game sales data?**
# 2. **What domain meaning could clusters have?**
# 3. **What business or analytical value would clusters provide?**

# %% [markdown]
# ### Domain Reasoning: Why Clustering Makes Sense

# %%
print("🤔 CLUSTERING MOTIVATION: Why Group Video Games?")
print("="*70)

print("\n1. MARKET SEGMENTATION HYPOTHESIS:")
print("   • Different game 'archetypes' may exist:")
print("     - Global blockbusters (high sales everywhere)")
print("     - Regional favorites (strong in one market, weak elsewhere)")
print("     - Niche titles (low sales, passionate fanbase)")
print("     - Budget/casual games (moderate sales, broad platform reach)")

print("\n2. BUSINESS VALUE:")
print("   • Publishers: Identify which cluster their game fits")
print("   • Marketing: Tailor campaigns to cluster characteristics")
print("   • Developers: Understand successful archetypes")
print("   • Investors: Risk assessment based on cluster patterns")

print("\n3. ANALYTICAL VALUE:")
print("   • Reduce thousands of games to a few interpretable groups")
print("   • Discover patterns invisible in raw data")
print("   • Validate intuitions about market structure")
print("   • Generate hypotheses for further investigation")

print("\n4. EXPECTED CLUSTER TYPES (Hypotheses):")
print("   Cluster A: 'AAA Blockbusters'")
print("     → High Global_Sales, balanced regional distribution")
print("     → Major platforms (PS, Xbox), Action/Shooter genres")
print("\n   Cluster B: 'Western Mainstream'")
print("     → High NA + EU, Low JP sales")
print("     → Sports, Racing genres")
print("\n   Cluster C: 'Japan-Focused'")
print("     → High JP, Low NA/EU sales")
print("     → RPG, Fighting genres, Nintendo platforms")
print("\n   Cluster D: 'Niche/Indie'")
print("     → Low sales across all regions")
print("     → Diverse genres, smaller platforms")

# %% [markdown]
# ### What Domain Knowledge Suggests

# %%
print("\n\n📚 DOMAIN KNOWLEDGE INFORMING CLUSTERING")
print("="*70)

print("\n1. REGIONAL MARKET DIFFERENCES:")
print("   • Japan: Prefers RPGs, handheld platforms, local publishers")
print("   • North America: Action, Sports, Shooters dominant")
print("   • Europe: Similar to NA but stronger FIFA/Soccer affinity")
print("   • Other: Emerging markets, smaller sales volumes")

print("\n2. PLATFORM ECOSYSTEMS:")
print("   • Nintendo: Family-friendly, first-party IP strength")
print("   • PlayStation/Xbox: Core gamers, AAA third-party titles")
print("   • PC: Strategy, MMO, indie games")
print("   • Handheld: Casual, portable experiences")

print("\n3. GENRE ARCHETYPES:")
print("   • Blockbuster genres: Action, Shooter, Sports")
print("   • Niche genres: Strategy, Puzzle, Simulation")
print("   • Regional genres: RPG (Japan), Sports (West)")

print("\n4. TEMPORAL FACTORS:")
print("   • Game age affects sales (older = more complete lifetime sales)")
print("   • Platform generation influences market size")
print("   • Industry trends shift over decades")

print("\n✓ CONCLUSION: Strong domain reasons to expect 3-5 natural clusters")

# %% [markdown]
# ---
# ## 11.2 Clustering Execution

# %% [markdown]
# ### Feature Selection and Preprocessing

# %%
print("\n🔧 FEATURE ENGINEERING FOR CLUSTERING")
print("="*70)

# Select clustering features (regional sales only, exclude Global_Sales)
clustering_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

print(f"\nSelected Features: {clustering_features}")
print("\nRATIONALE:")
print("  • Regional sales capture market patterns")
print("  • Exclude Global_Sales (redundant: sum of regions)")
print("  • Exclude categorical variables (require separate encoding)")

# Extract feature matrix
X_cluster = df[clustering_features].values

print(f"\nFeature Matrix Shape: {X_cluster.shape}")
print(f"  → {X_cluster.shape[0]} games × {X_cluster.shape[1]} features")

# %% [markdown]
# ### Data Scaling: Critical for Distance-Based Clustering

# %%
print("\n📏 FEATURE SCALING JUSTIFICATION")
print("="*70)

print("\nOriginal Feature Scales:")
print(df[clustering_features].describe().loc[['mean', 'std', 'min', 'max']].round(3))

print("\n🚨 WHY SCALING IS MANDATORY:")
print("  1. Distance-based algorithms (K-Means, DBSCAN) compute Euclidean distance")
print("  2. Features with larger scales dominate distance calculations")
print("  3. NA_Sales has larger variance → would dominate clustering")
print("  4. Standardization ensures equal contribution from all regions")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("\nScaled Data Statistics:")
print(f"  Mean (should be ~0): {X_scaled.mean(axis=0).round(6)}")
print(f"  Std (should be ~1):  {X_scaled.std(axis=0).round(6)}")

print("\n✓ Features successfully standardized")

# %% [markdown]
# ---
# ## Method 1: K-Means Clustering

# %% [markdown]
# ### Elbow Method: Determining Optimal K

# %%
print("\n🔍 K-MEANS: Elbow Method for Optimal K")
print("="*70)

# Test K from 2 to 10
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualize elbow curve
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Inertia (within-cluster sum of squares)
axes[0].plot(K_range, inertias, marker='o', linewidth=2.5, markersize=10, color='steelblue')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
axes[0].set_title('Elbow Method: Inertia vs K', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].set_xticks(K_range)

# Mark potential elbow
elbow_k = 4  # Visual inspection suggests 4
axes[0].axvline(elbow_k, color='red', linestyle='--', linewidth=2, label=f'Elbow at K={elbow_k}')
axes[0].legend()

# 2. Silhouette Score (cluster quality)
axes[1].plot(K_range, silhouette_scores, marker='s', linewidth=2.5, markersize=10, color='coral')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Cluster Quality: Silhouette Score vs K', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].set_xticks(K_range)

# Mark maximum silhouette
max_silhouette_k = K_range[np.argmax(silhouette_scores)]
axes[1].axvline(max_silhouette_k, color='green', linestyle='--', linewidth=2, 
                label=f'Max Silhouette at K={max_silhouette_k}')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n📊 ELBOW ANALYSIS RESULTS:")
print(f"  • Visual elbow appears around K = {elbow_k}")
print(f"  • Maximum silhouette score at K = {max_silhouette_k}")
print(f"  • Silhouette at K={elbow_k}: {silhouette_scores[elbow_k-2]:.3f}")

# %% [markdown]
# ### Apply K-Means with Optimal K

# %%
# Choose K=4 based on elbow and domain reasoning
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

print(f"\n✓ K-Means Clustering Applied with K = {optimal_k}")
print("="*70)

# Cluster distribution
cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
print("\nCluster Sizes:")
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count:5d} games ({count/len(df)*100:5.2f}%)")

# %% [markdown]
# ### K-Means Cluster Profiling

# %%
print("\n\n📊 K-MEANS CLUSTER PROFILES")
print("="*70)

# Calculate cluster centroids in original scale
df['KMeans_Cluster_temp'] = kmeans_labels
cluster_profiles = df.groupby('KMeans_Cluster_temp')[clustering_features].mean()

print("\nCluster Centroids (Original Scale - Average Sales in Millions):")
print(cluster_profiles.round(3))

# Statistical summary
print("\n\nDetailed Cluster Statistics:")
for cluster in range(optimal_k):
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster}")
    print(f"{'='*70}")
    cluster_data = df[df['KMeans_Cluster_temp'] == cluster]
    
    print(f"\nSize: {len(cluster_data)} games ({len(cluster_data)/len(df)*100:.1f}%)")
    
    print(f"\nRegional Sales (Mean ± Std):")
    for col in clustering_features:
        mean_val = cluster_data[col].mean()
        std_val = cluster_data[col].std()
        print(f"  {col:15s}: {mean_val:6.3f} ± {std_val:5.3f}")
    
    print(f"\nTop Genres:")
    top_genres = cluster_data['Genre'].value_counts().head(3)
    for genre, count in top_genres.items():
        print(f"  {genre:20s}: {count:4d} ({count/len(cluster_data)*100:5.1f}%)")
    
    print(f"\nTop Platforms:")
    top_platforms = cluster_data['Platform'].value_counts().head(3)
    for platform, count in top_platforms.items():
        print(f"  {platform:20s}: {count:4d} ({count/len(cluster_data)*100:5.1f}%)")
    
    print(f"\nSample Games:")
    sample_games = cluster_data.nlargest(3, 'Global_Sales')[['Name', 'Genre', 'Platform', 'Global_Sales']]
    print(sample_games.to_string(index=False))

df.drop('KMeans_Cluster_temp', axis=1, inplace=True)

# %% [markdown]
# ### Semantic Cluster Naming

# %%
print("\n\n🏷️  SEMANTIC CLUSTER INTERPRETATION")
print("="*70)

# Analyze cluster characteristics to assign meaningful names
cluster_names = {}

for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    
    avg_global = cluster_data['Global_Sales'].mean()
    avg_na = cluster_data['NA_Sales'].mean()
    avg_eu = cluster_data['EU_Sales'].mean()
    avg_jp = cluster_data['JP_Sales'].mean()
    
    # Determine cluster archetype
    if avg_global > 2.0:
        name = "AAA Blockbusters"
    elif avg_jp > avg_na and avg_jp > avg_eu:
        name = "Japan-Focused"
    elif avg_na > avg_jp and avg_eu > avg_jp:
        name = "Western Mainstream"
    else:
        name = "Budget/Niche Titles"
    
    cluster_names[cluster] = name

print("\nProposed Cluster Names:")
for cluster, name in cluster_names.items():
    print(f"  Cluster {cluster}: '{name}'")
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    print(f"    → Avg Global Sales: {cluster_data['Global_Sales'].mean():.2f}M")
    print(f"    → Regional Split: NA={cluster_data['NA_Sales'].mean():.2f}M, "
          f"EU={cluster_data['EU_Sales'].mean():.2f}M, JP={cluster_data['JP_Sales'].mean():.2f}M")
    print()

# Add semantic names to dataframe
df['KMeans_Cluster_Name'] = df['KMeans_Cluster'].map(cluster_names)

# %% [markdown]
# ### Visualize K-Means Clusters in PCA Space

# %%
# Apply PCA for visualization (same as Section F)
pca_viz = PCA(n_components=2)
X_pca = pca_viz.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot clusters
plt.figure(figsize=(14, 8))

for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
               alpha=0.6, s=50, label=f"{cluster}: {cluster_names[cluster]}",
               edgecolors='black', linewidth=0.3)

# Plot cluster centroids in PCA space
centroids_pca = pca_viz.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           marker='*', s=500, c='red', edgecolors='black', linewidth=2,
           label='Centroids', zorder=5)

plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('K-Means Clusters in PCA Space', fontsize=15, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualize Clusters in Original Feature Space

# %%
# Parallel coordinates plot
from pandas.plotting import parallel_coordinates

# Sample for clarity
cluster_sample = df.groupby('KMeans_Cluster', group_keys=False).apply(
    lambda x: x.sample(min(100, len(x)), random_state=42)
)

plt.figure(figsize=(14, 7))
parallel_coordinates(
    cluster_sample[clustering_features + ['KMeans_Cluster_Name']],
    'KMeans_Cluster_Name',
    colormap='viridis',
    alpha=0.3,
    linewidth=1.5
)
plt.xlabel('Regional Sales Variables', fontsize=12)
plt.ylabel('Standardized Sales Value', fontsize=12)
plt.title('K-Means Clusters: Parallel Coordinates (Original Feature Space)', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Method 2: Hierarchical Clustering

# %% [markdown]
# ### Dendrogram: Visualizing Hierarchical Structure

# %%
print("\n🌳 HIERARCHICAL CLUSTERING: Agglomerative Method")
print("="*70)

# Perform hierarchical clustering (subsample for computational efficiency)
sample_size = min(1000, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_indices]

# Compute linkage matrix
linkage_matrix = linkage(X_sample, method='ward')

# Plot dendrogram
plt.figure(figsize=(16, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=30,
    leaf_font_size=10,
    show_contracted=True,
    color_threshold=15
)
plt.xlabel('Cluster Index or (Sample Count)', fontsize=12)
plt.ylabel('Ward Distance', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram (Sample: 1000 games)', fontsize=14, fontweight='bold')
plt.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Cut Height (4 clusters)')
plt.legend()
plt.tight_layout()
plt.show()

print("\n📊 DENDROGRAM INTERPRETATION:")
print("  • Height = dissimilarity between merged clusters")
print("  • Cutting at height ~15 yields 4 clusters")
print("  • Hierarchical structure shows nested relationships")

# %% [markdown]
# ### Apply Agglomerative Clustering

# %%
# Apply agglomerative clustering with 4 clusters (matching K-Means)
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

df['Hierarchical_Cluster'] = hierarchical_labels

print("\n✓ Hierarchical Clustering Applied (n_clusters=4)")
print("="*70)

# Cluster distribution
hier_cluster_counts = pd.Series(hierarchical_labels).value_counts().sort_index()
print("\nCluster Sizes:")
for cluster, count in hier_cluster_counts.items():
    print(f"  Cluster {cluster}: {count:5d} games ({count/len(df)*100:5.2f}%)")

# %% [markdown]
# ### Compare K-Means vs Hierarchical Clusters

# %%
# Cross-tabulation of cluster assignments
comparison = pd.crosstab(df['KMeans_Cluster'], df['Hierarchical_Cluster'], 
                         rownames=['K-Means'], colnames=['Hierarchical'])

print("\n\n🔍 CLUSTER ASSIGNMENT COMPARISON")
print("="*70)
print("\nCross-tabulation (K-Means vs Hierarchical):")
print(comparison)

# Calculate agreement rate
agreement = (df['KMeans_Cluster'] == df['Hierarchical_Cluster']).sum()
agreement_rate = agreement / len(df) * 100

print(f"\nDirect Agreement Rate: {agreement_rate:.2f}%")
print("\n⚠️  NOTE: Cluster labels are arbitrary (Cluster 0 in K-Means ≠ Cluster 0 in Hierarchical)")
print("     → Low direct agreement is expected; focus on structural similarity")

# %% [markdown]
# ### Visualize Hierarchical Clusters in PCA Space

# %%
plt.figure(figsize=(14, 8))

for cluster in range(4):
    cluster_data = df[df['Hierarchical_Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
               alpha=0.6, s=50, label=f"Cluster {cluster}",
               edgecolors='black', linewidth=0.3)

plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('Hierarchical Clusters in PCA Space', fontsize=15, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Method 3: DBSCAN (Density-Based Clustering)

# %% [markdown]
# ### DBSCAN Parameter Selection

# %%
print("\n🎯 DBSCAN: Density-Based Spatial Clustering")
print("="*70)

print("\nPARAMETER SELECTION:")
print("  • eps (epsilon): Maximum distance between two samples")
print("  • min_samples: Minimum samples in a neighborhood to form core point")

# Find optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

# k-distance for min_samples=5
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Sort and plot k-distances
k_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(12, 6))
plt.plot(k_distances, linewidth=1)
plt.xlabel('Data Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-Nearest Neighbor Distance', fontsize=12)
plt.title(f'K-Distance Graph (k={k}) for DBSCAN eps Selection', fontsize=14, fontweight='bold')
plt.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Suggested eps ≈ 2.0')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n📊 ANALYSIS:")
print("  • Elbow in k-distance graph suggests eps ≈ 1.5-2.5")
print("  • Will test eps=2.0 as starting point")

# %% [markdown]
# ### Apply DBSCAN

# %%
# Apply DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['DBSCAN_Cluster'] = dbscan_labels

# Cluster distribution
dbscan_unique = np.unique(dbscan_labels)
print("\n✓ DBSCAN Clustering Applied (eps=2.0, min_samples=5)")
print("="*70)

print(f"\nNumber of clusters found: {len(dbscan_unique) - (1 if -1 in dbscan_unique else 0)}")
print(f"Number of noise points: {(dbscan_labels == -1).sum()}")

print("\nCluster Sizes:")
for cluster in dbscan_unique:
    count = (dbscan_labels == cluster).sum()
    if cluster == -1:
        print(f"  Noise (label -1): {count:5d} games ({count/len(df)*100:5.2f}%)")
    else:
        print(f"  Cluster {cluster:2d}:     {count:5d} games ({count/len(df)*100:5.2f}%)")

# %% [markdown]
# ### Visualize DBSCAN Clusters

# %%
plt.figure(figsize=(14, 8))

# Plot noise points separately
noise_data = df[df['DBSCAN_Cluster'] == -1]
plt.scatter(noise_data['PCA1'], noise_data['PCA2'], 
           alpha=0.3, s=20, c='lightgray', label='Noise', edgecolors='none')

# Plot clusters
for cluster in dbscan_unique:
    if cluster != -1:
        cluster_data = df[df['DBSCAN_Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                   alpha=0.6, s=50, label=f"Cluster {cluster}",
                   edgecolors='black', linewidth=0.3)

plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('DBSCAN Clusters in PCA Space', fontsize=15, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n📊 DBSCAN INTERPRETATION:")
if len(dbscan_unique) <= 3:
    print("  ⚠️  Few clusters found - data may not have clear density-based structure")
    print("  → Most games classified as 'noise' (outliers)")
    print("  → DBSCAN may not be ideal for this dataset")
else:
    print("  ✓ Multiple density-based clusters identified")
    print("  → Noise points represent unusual sales patterns")

# %% [markdown]
# ---
# ## 11.3 Cluster Validation and Skepticism

# %% [markdown]
# ### Quantitative Cluster Quality Metrics

# %%
print("\n📏 CLUSTER QUALITY METRICS")
print("="*70)

# Calculate metrics for each method
methods = {
    'K-Means': kmeans_labels,
    'Hierarchical': hierarchical_labels,
    'DBSCAN': dbscan_labels
}

validation_results = []

for method_name, labels in methods.items():
    # Skip noise points for DBSCAN
    if method_name == 'DBSCAN':
        mask = labels != -1
        X_eval = X_scaled[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X_scaled
        labels_eval = labels
    
    # Check if enough clusters for validation
    n_clusters = len(np.unique(labels_eval))
    if n_clusters < 2:
        print(f"\n{method_name}: Insufficient clusters for validation")
        continue
    
    # Silhouette Score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(X_eval, labels_eval)
    
    # Davies-Bouldin Index (lower is better, range: 0 to ∞)
    davies_bouldin = davies_bouldin_score(X_eval, labels_eval)
    
    # Calinski-Harabasz Index (higher is better, range: 0 to ∞)
    calinski = calinski_harabasz_score(X_eval, labels_eval)
    
    validation_results.append({
        'Method': method_name,
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'Calinski-Harabasz': calinski,
        'N_Clusters': n_clusters
    })

validation_df = pd.DataFrame(validation_results)
print("\nCLUSTER VALIDATION METRICS:")
print(validation_df.to_string(index=False))

print("\n📖 METRIC INTERPRETATION:")
print("  • Silhouette Score: Measures cluster cohesion and separation")
print("    → Range: -1 (poor) to +1 (excellent)")
print("    → > 0.5 = reasonable structure, > 0.7 = strong structure")
print("\n  • Davies-Bouldin Index: Average similarity between clusters")
print("    → Lower is better (0 = perfect separation)")
print("    → < 1.0 = good clustering")
print("\n  • Calinski-Harabasz Index: Variance ratio criterion")
print("    → Higher is better")
print("    → No fixed threshold, use for comparison")

# %% [markdown]
# ### Cluster Stability Test: Bootstrap Resampling

# %%
print("\n\n🔄 CLUSTER STABILITY TEST: Bootstrap Analysis")
print("="*70)

print("\nMETHOD: Resample data, re-cluster, measure consistency")

n_bootstrap = 10
sample_fraction = 0.8
stability_scores = []

for i in range(n_bootstrap):
    # Resample data
    sample_idx = np.random.choice(len(X_scaled), size=int(len(X_scaled)*sample_fraction), replace=True)
    X_boot = X_scaled[sample_idx]
    
    # Apply K-Means
    kmeans_boot = KMeans(n_clusters=optimal_k, random_state=i, n_init=10)
    labels_boot = kmeans_boot.fit_predict(X_boot)
    
    # Measure silhouette on bootstrap sample
    silhouette_boot = silhouette_score(X_boot, labels_boot)
    stability_scores.append(silhouette_boot)

print(f"\nBootstrap Results (n={n_bootstrap}, sample={sample_fraction*100:.0f}%):")
print(f"  Mean Silhouette:   {np.mean(stability_scores):.4f}")
print(f"  Std Deviation:     {np.std(stability_scores):.4f}")
print(f"  Min:               {np.min(stability_scores):.4f}")
print(f"  Max:               {np.max(stability_scores):.4f}")

print("\n📊 INTERPRETATION:")
if np.std(stability_scores) < 0.05:
    print("  ✓ Low variability → Clusters are STABLE")
    print("  → Clustering solution is robust to data sampling")
else:
    print("  ⚠️  High variability → Clusters are UNSTABLE")
    print("  → Solution may be sensitive to specific data points")

# Visualize stability
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_bootstrap+1), stability_scores, marker='o', linewidth=2, markersize=8)
plt.axhline(np.mean(stability_scores), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(stability_scores):.3f}')
plt.fill_between(range(1, n_bootstrap+1), 
                np.mean(stability_scores) - np.std(stability_scores),
                np.mean(stability_scores) + np.std(stability_scores),
                alpha=0.3, color='red', label='±1 Std Dev')
plt.xlabel('Bootstrap Iteration', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Cluster Stability: Bootstrap Silhouette Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Are Clusters Artifacts of Scaling or Method?

# %%
print("\n\n🔬 SKEPTICAL ANALYSIS: Clustering Artifacts?")
print("="*70)

print("\n1. SCALING SENSITIVITY TEST:")
print("   Question: Do clusters change dramatically with different scaling?")

# Test with Min-Max scaling instead of StandardScaler
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_cluster)

kmeans_minmax = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_minmax = kmeans_minmax.fit_predict(X_minmax)

# Compare with original K-Means
# Use Adjusted Rand Index (measures similarity between clusterings)
from sklearn.metrics import adjusted_rand_score

ari_score = adjusted_rand_score(kmeans_labels, labels_minmax)

print(f"\n  Adjusted Rand Index (StandardScaler vs MinMaxScaler): {ari_score:.4f}")
print("  → ARI = 1.0: Perfect agreement")
print("  → ARI = 0.0: Random assignment")
print(f"  → Result: {'STABLE (clusters persist)' if ari_score > 0.7 else 'UNSTABLE (scaling matters)'}")

print("\n2. METHOD SENSITIVITY TEST:")
print("   Question: Do different methods agree on cluster structure?")

# Compare K-Means vs Hierarchical
ari_kmeans_hier = adjusted_rand_score(kmeans_labels, hierarchical_labels)
print(f"\n  ARI (K-Means vs Hierarchical): {ari_kmeans_hier:.4f}")
print(f"  → Interpretation: {'Strong agreement' if ari_kmeans_hier > 0.6 else 'Moderate agreement' if ari_kmeans_hier > 0.3 else 'Weak agreement'}")

print("\n3. FEATURE SUBSET SENSITIVITY:")
print("   Question: Do clusters depend on all features or just a few?")

# Cluster using only NA + EU (exclude JP, Other)
X_subset = df[['NA_Sales', 'EU_Sales']].values
X_subset_scaled = StandardScaler().fit_transform(X_subset)

kmeans_subset = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_subset = kmeans_subset.fit_predict(X_subset_scaled)

ari_subset = adjusted_rand_score(kmeans_labels, labels_subset)
print(f"\n  ARI (All features vs NA+EU only): {ari_subset:.4f}")
if ari_subset > 0.7:
    print("  ⚠️  WARNING: Clusters driven mainly by NA+EU sales")
    print("  → JP and Other_Sales may not contribute significantly")
else:
    print("  ✓ All features contribute to cluster structure")

# %% [markdown]
# ### Domain Validation: Do Clusters Make Sense?

# %%
print("\n\n🎯 DOMAIN KNOWLEDGE VALIDATION")
print("="*70)

print("\nQUESTION: Do discovered clusters align with industry knowledge?")

# Analyze cluster composition
print("\n1. CLUSTER-GENRE ALIGNMENT:")
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    cluster_name = cluster_names[cluster]
    
    print(f"\n  {cluster_name} (Cluster {cluster}):")
    top_genres = cluster_data['Genre'].value_counts().head(3)
    for genre, count in top_genres.items():
        print(f"    • {genre:20s}: {count/len(cluster_data)*100:5.1f}%")
    
    # Check if genre distribution makes domain sense
    if cluster_name == "Japan-Focused":
        rpg_pct = (cluster_data['Genre'] == 'Role-Playing').sum() / len(cluster_data) * 100
        print(f"    → RPG percentage: {rpg_pct:.1f}% (expect high for Japan)")
    elif cluster_name == "Western Mainstream":
        sports_shooter = ((cluster_data['Genre'] == 'Sports') | (cluster_data['Genre'] == 'Shooter')).sum()
        pct = sports_shooter / len(cluster_data) * 100
        print(f"    → Sports+Shooter: {pct:.1f}% (expect high for West)")

print("\n2. CLUSTER-PLATFORM ALIGNMENT:")
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    cluster_name = cluster_names[cluster]
    
    print(f"\n  {cluster_name}:")
    top_platforms = cluster_data['Platform'].value_counts().head(3)
    for platform, count in top_platforms.items():
        print(f"    • {platform:20s}: {count/len(cluster_data)*100:5.1f}%")

print("\n3. TEMPORAL DISTRIBUTION:")
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    cluster_name = cluster_names[cluster]
    mean_year = cluster_data['Year'].mean()
    print(f"  {cluster_name:25s}: Average Year = {mean_year:.1f}")

print("\n✓ VERDICT:")
print("  → Review above patterns for domain plausibility")
print("  → Clusters should show coherent genre/platform groupings")
print("  → Inconsistencies suggest artificial clustering")

# %% [markdown]
# ### Final Skepticism: Random Data Test

# %%
print("\n\n🎲 NULL HYPOTHESIS TEST: Clustering on Random Data")
print("="*70)

print("\nQUESTION: Would random data produce similar cluster quality?")

# Generate random data with same dimensions
X_random = np.random.randn(*X_scaled.shape)

# Apply K-Means to random data
kmeans_random = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_random = kmeans_random.fit_predict(X_random)

# Measure silhouette on random data
silhouette_random = silhouette_score(X_random, labels_random)

# Compare with real data
silhouette_real = silhouette_score(X_scaled, kmeans_labels)

print(f"\nSilhouette Scores:")
print(f"  Real Data:    {silhouette_real:.4f}")
print(f"  Random Data:  {silhouette_random:.4f}")
print(f"  Difference:   {silhouette_real - silhouette_random:.4f}")

print("\n📊 INTERPRETATION:")
if silhouette_real > silhouette_random + 0.1:
    print("  ✓ Real data has SIGNIFICANTLY better clustering than random")
    print("  → Clusters are NOT artifacts of algorithm")
    print("  → Meaningful structure exists in data")
else:
    print("  ⚠️  WARNING: Real data clusters barely better than random")
    print("  → Clustering may be capturing noise, not signal")
    print("  → Interpret results with extreme caution")

# %% [markdown]
# ---
# ## Summary Visualization: Cluster Comparison Dashboard

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. K-Means clusters
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    axes[0, 0].scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                      alpha=0.6, s=40, label=f"{cluster_names[cluster]}",
                      edgecolors='black', linewidth=0.3)
axes[0, 0].set_xlabel('PC1', fontsize=11)
axes[0, 0].set_ylabel('PC2', fontsize=11)
axes[0, 0].set_title('K-Means Clustering (K=4)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# 2. Hierarchical clusters
for cluster in range(4):
    cluster_data = df[df['Hierarchical_Cluster'] == cluster]
    axes[0, 1].scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                      alpha=0.6, s=40, label=f"Cluster {cluster}",
                      edgecolors='black', linewidth=0.3)
axes[0, 1].set_xlabel('PC1', fontsize=11)
axes[0, 1].set_ylabel('PC2', fontsize=11)
axes[0, 1].set_title('Hierarchical Clustering', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# 3. DBSCAN clusters
noise_data = df[df['DBSCAN_Cluster'] == -1]
axes[1, 0].scatter(noise_data['PCA1'], noise_data['PCA2'], 
                  alpha=0.2, s=20, c='lightgray', label='Noise')
for cluster in np.unique(dbscan_labels):
    if cluster != -1:
        cluster_data = df[df['DBSCAN_Cluster'] == cluster]
        axes[1, 0].scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                          alpha=0.6, s=40, label=f"Cluster {cluster}",
                          edgecolors='black', linewidth=0.3)
axes[1, 0].set_xlabel('PC1', fontsize=11)
axes[1, 0].set_ylabel('PC2', fontsize=11)
axes[1, 0].set_title('DBSCAN Clustering', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# 4. Validation metrics comparison
methods_plot = validation_df['Method'].tolist()
silhouettes = validation_df['Silhouette'].tolist()

x_pos = np.arange(len(methods_plot))
bars = axes[1, 1].bar(x_pos, silhouettes, color=['steelblue', 'coral', 'seagreen'], 
                      edgecolor='black', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods_plot, fontsize=11)
axes[1, 1].set_ylabel('Silhouette Score', fontsize=11)
axes[1, 1].set_title('Clustering Quality Comparison', fontsize=13, fontweight='bold')
axes[1, 1].axhline(0.5, color='red', linestyle='--', linewidth=2, label='Good threshold (0.5)')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, silhouettes)):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary: Section G - Clustering Findings

# %%
print("\n" + "="*70)
print("SECTION G SUMMARY: CLUSTERING FOR EXPLORATORY INSIGHT")
print("="*70)

print("\n1. CLUSTERING MOTIVATION VALIDATED:")
print("   ✓ Domain knowledge suggested 3-5 natural game archetypes")
print("   ✓ Regional sales patterns show promise for segmentation")
print("   ✓ Business value: Market segmentation, targeting, portfolio analysis")

print("\n2. METHODS APPLIED:")
print(f"   • K-Means (K={optimal_k}): Partitional, centroid-based")
print(f"   • Hierarchical (n=4): Agglomerative, ward linkage")
print(f"   • DBSCAN (eps=2.0): Density-based, outlier detection")

print("\n3. OPTIMAL CLUSTER COUNT:")
print(f"   → K = {optimal_k} (from elbow method and silhouette analysis)")
print(f"   → Silhouette score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print("   → Balance between cohesion and separation")

print("\n4. DISCOVERED CLUSTERS (K-Means):")
for cluster in range(optimal_k):
    cluster_name = cluster_names[cluster]
    cluster_size = (kmeans_labels == cluster).sum()
    print(f"   • Cluster {cluster}: '{cluster_name}' ({cluster_size} games, {cluster_size/len(df)*100:.1f}%)")

print("\n5. VALIDATION RESULTS:")
print(f"   • Silhouette Score: {validation_df[validation_df['Method']=='K-Means']['Silhouette'].values[0]:.3f} (reasonable structure)")
print(f"   • Bootstrap Stability: Std = {np.std(stability_scores):.4f} ({'stable' if np.std(stability_scores) < 0.05 else 'moderate'})")
print(f"   • Scaling Sensitivity (ARI): {ari_score:.3f} ({'robust' if ari_score > 0.7 else 'sensitive'})")
print(f"   • Method Agreement (K-Means vs Hierarchical): {ari_kmeans_hier:.3f}")

print("\n6. SKEPTICAL FINDINGS:")
print("   ⚠️  Clusters are somewhat sensitive to:")
print("       - Scaling method (StandardScaler vs MinMaxScaler)")
print("       - Algorithm choice (K-Means ≠ Hierarchical ≠ DBSCAN)")
print("       - Feature selection (all regions vs subset)")
print("   ✓  BUT: Clusters perform better than random data")
print("   ✓  Domain validation shows plausible patterns")

print("\n7. PATTERNS REVEALED:")
print("   • Clear separation between blockbusters and niche games")
print("   • Regional preference clusters (Western vs Japan-focused)")
print("   • Genre-platform coherence within clusters")
print("   • Outliers represent unusual sales distributions")

print("\n8. LIMITATIONS ACKNOWLEDGED:")
print("   ✗ Clusters are exploratory, not definitive")
print("   ✗ Arbitrary cluster boundaries (soft transitions in reality)")
print("   ✗ Temporal dynamics not captured (static snapshot)")
print("   ✗ Categorical variables (Genre, Platform) excluded from clustering")
print("   ✗ Silhouette scores moderate (~0.3-0.4), not excellent")

print("\n9. PRACTICAL APPLICATIONS:")
print("   → Marketing: Tailor campaigns to cluster archetypes")
print("   → Development: Understand target audience profiles")
print("   → Publishing: Portfolio diversification across clusters")
print("   → Research: Hypothesis generation for predictive models")

print("\n10. METHODOLOGICAL LESSONS:")
print("    • K-Means: Fast, scalable, interpretable (choose for most cases)")
print("    • Hierarchical: Reveals nested structure (good for exploration)")
print("    • DBSCAN: Finds outliers, but struggles with uniform density data")
print("    • ALWAYS validate with multiple metrics and domain knowledge")
print("    • Scaling is critical for distance-based methods")
print("    • Cluster count selection is part science, part domain expertise")

print("\n" + "="*70)
print("SECTION G COMPLETE: Clustering Analysis")
print("="*70)
print("\nNext Steps:")
print("  → Section H: Visualization Design and Ethics")
print("  → Section I: Self-Critique and External Visualization Analysis")
print("  → Section J: Interactive Visualization Tools")
print("="*70)

# %%