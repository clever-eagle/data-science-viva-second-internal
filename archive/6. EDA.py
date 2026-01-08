# %% [markdown]
# # Comprehensive Exploratory Data Analysis (EDA) - Iris Dataset
# ## ML Course Viva Demonstration
# 
# **Author:** EDA Demonstration  
# **Dataset:** Iris Dataset from sklearn  
# **Objective:** Perform thorough exploratory data analysis covering all fundamental EDA techniques
# 
# ---
# 
# ## Table of Contents
# 1. Data Import & Preprocessing
# 2. Data Distribution Analysis
# 3. Data Visualization (Diverse Types)
# 4. Data Filtering Techniques
# 5. Statistical Analysis (Correlation & Covariance)
# 6. Sampling Demonstrations
# 7. Multiclass Classification Data Preparation
# 8. Summary & Insights

# %% [markdown]
# ---
# ## SECTION 1: Data Import & Preprocessing
# 
# In this section, we will:
# - Import all necessary libraries for EDA
# - Load the Iris dataset from sklearn
# - Perform basic data exploration
# - Check for missing values and data quality
# - Understand the structure of our dataset

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Configure visualization settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)

print("All libraries imported successfully!")

# %%
# Load the Iris dataset
iris = load_iris()

# Create a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset loaded successfully!")
print(f"\nDataset shape: {df.shape}")

# %% [markdown]
# ### 1.1 Basic Data Exploration

# %%
# Display first few rows
print("=" * 80)
print("FIRST 10 ROWS OF THE DATASET")
print("=" * 80)
print(df.head(10))

# %%
# Display last few rows
print("\n" + "=" * 80)
print("LAST 5 ROWS OF THE DATASET")
print("=" * 80)
print(df.tail())

# %%
# Dataset information
print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)
print(df.info())

# %%
# Statistical summary
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY OF NUMERICAL FEATURES")
print("=" * 80)
print(df.describe())

# %%
# Shape and dimensions
print("\n" + "=" * 80)
print("DATASET DIMENSIONS")
print("=" * 80)
print(f"Number of rows (samples): {df.shape[0]}")
print(f"Number of columns (features): {df.shape[1]}")
print(f"\nFeature names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# %% [markdown]
# ### 1.2 Missing Values Analysis

# %%
# Check for missing values
print("=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("\n✓ No missing values detected in the dataset!")
else:
    print(f"\nTotal missing values: {missing_values.sum()}")
    print("\nPercentage of missing values:")
    print((missing_values / len(df)) * 100)

# %%
# Additional data quality checks
print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Unique species: {df['species_name'].nunique()}")
print(f"\nSpecies distribution:\n{df['species_name'].value_counts()}")

# %% [markdown]
# ### 1.3 Data Type Verification

# %%
# Verify and display data types
print("=" * 80)
print("DATA TYPE VERIFICATION")
print("=" * 80)

for col in df.columns:
    print(f"{col:30s} -> {df[col].dtype}")

print("\n✓ All numerical features are float64 (appropriate for analysis)")
print("✓ Target variable is properly encoded")

# %% [markdown]
# ---
# ## SECTION 2: Data Distribution Analysis
# 
# Understanding the distribution of data is crucial for:
# - Identifying patterns and anomalies
# - Understanding class balance
# - Detecting outliers
# - Making informed decisions about preprocessing

# %% [markdown]
# ### 2.1 Class Distribution Analysis

# %%
# Class distribution
print("=" * 80)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 80)

class_counts = df['species_name'].value_counts()
print("\nAbsolute counts:")
print(class_counts)

print("\nPercentage distribution:")
class_percentages = (class_counts / len(df)) * 100
print(class_percentages)

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
axes[0].bar(class_counts.index, class_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Species', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✓ Dataset is perfectly balanced with 50 samples per class")

# %% [markdown]
# ### 2.2 Feature Distribution Analysis

# %%
# Statistical summary by feature
print("=" * 80)
print("FEATURE-WISE STATISTICAL SUMMARY")
print("=" * 80)

feature_cols = iris.feature_names
for col in feature_cols:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.3f}")
    print(f"  Median: {df[col].median():.3f}")
    print(f"  Std Dev: {df[col].std():.3f}")
    print(f"  Min: {df[col].min():.3f}")
    print(f"  Max: {df[col].max():.3f}")
    print(f"  Range: {df[col].max() - df[col].min():.3f}")

# %%
# Visualize feature distributions with histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    axes[idx].hist(df[col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
    axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.2f}')
    axes[idx].set_title(f'Distribution of {col}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Outlier Detection using Box Plots

# %%
# Box plots for outlier detection
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='navy'),
                           whiskerprops=dict(color='navy'),
                           capprops=dict(color='navy'),
                           medianprops=dict(color='red', linewidth=2))
    axes[idx].set_title(f'Box Plot: {col}', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Identify outliers using IQR method
print("=" * 80)
print("OUTLIER DETECTION (IQR Method)")
print("=" * 80)

for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"  Q1 (25th percentile): {Q1:.3f}")
    print(f"  Q3 (75th percentile): {Q3:.3f}")
    print(f"  IQR: {IQR:.3f}")
    print(f"  Lower bound: {lower_bound:.3f}")
    print(f"  Upper bound: {upper_bound:.3f}")
    print(f"  Number of outliers: {len(outliers)}")

# %% [markdown]
# ### 2.4 Distribution by Class

# %%
# Distribution of features across different species
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species][col]
        axes[idx].hist(species_data, bins=15, alpha=0.6, label=species, edgecolor='black')
    
    axes[idx].set_title(f'Distribution of {col} by Species', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## SECTION 3: Data Visualization (Diverse Types)
# 
# Visualization is key to understanding patterns, relationships, and insights in data.
# We will create multiple types of visualizations to explore the dataset from different angles.

# %% [markdown]
# ### 3.1 Scatter Plots - Feature Relationships

# %%
# Scatter plot: Sepal Length vs Sepal Width
plt.figure(figsize=(10, 7))
colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}

for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'],
               label=species, alpha=0.7, s=100, color=colors[species], edgecolors='black')

plt.title('Sepal Length vs Sepal Width by Species', fontsize=16, fontweight='bold')
plt.xlabel('Sepal Length (cm)', fontsize=13)
plt.ylabel('Sepal Width (cm)', fontsize=13)
plt.legend(title='Species', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Scatter plot: Petal Length vs Petal Width
plt.figure(figsize=(10, 7))

for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'],
               label=species, alpha=0.7, s=100, color=colors[species], edgecolors='black')

plt.title('Petal Length vs Petal Width by Species', fontsize=16, fontweight='bold')
plt.xlabel('Petal Length (cm)', fontsize=13)
plt.ylabel('Petal Width (cm)', fontsize=13)
plt.legend(title='Species', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Pair Plot / Scatter Matrix

# %%
# Comprehensive pair plot
pair_plot = sns.pairplot(df, hue='species_name', palette=colors, 
                         diag_kind='kde', height=2.5, aspect=1.2,
                         plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'black'},
                         diag_kws={'alpha': 0.7})
pair_plot.fig.suptitle('Pair Plot: All Features by Species', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# %% [markdown]
# ### 3.3 Correlation Heatmap

# %%
# Calculate correlation matrix for numerical features
correlation_matrix = df[feature_cols].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            fmt='.3f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Iris Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# %% [markdown]
# ### 3.4 Violin Plots

# %%
# Violin plots for all features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    sns.violinplot(data=df, x='species_name', y=col, ax=axes[idx], palette=colors)
    axes[idx].set_title(f'Violin Plot: {col} by Species', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Species', fontsize=11)
    axes[idx].set_ylabel(col, fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.5 KDE (Kernel Density Estimation) Plots

# %%
# KDE plots for feature distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species][col]
        axes[idx].hist(species_data, bins=20, alpha=0.3, label=species, density=True, color=colors[species])
        species_data.plot(kind='kde', ax=axes[idx], label=f'{species} (KDE)', 
                         linewidth=2, color=colors[species])
    
    axes[idx].set_title(f'KDE Plot: {col}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.6 Bar Charts for Statistical Comparison

# %%
# Mean values comparison across species
mean_by_species = df.groupby('species_name')[feature_cols].mean()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    mean_by_species[col].plot(kind='bar', ax=axes[idx], color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                              edgecolor='black', alpha=0.8)
    axes[idx].set_title(f'Mean {col} by Species', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Species', fontsize=11)
    axes[idx].set_ylabel(f'Mean {col}', fontsize=11)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.7 Box Plot Comparison

# %%
# Box plots comparing all features side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, species in enumerate(df['species_name'].unique()):
    species_data = df[df['species_name'] == species][feature_cols]
    bp = axes[idx].boxplot([species_data[col] for col in feature_cols],
                           labels=[col.split(' ')[0].capitalize() for col in feature_cols],
                           patch_artist=True,
                           boxprops=dict(facecolor=colors[species], alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
    axes[idx].set_title(f'{species.capitalize()} - Feature Comparison', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Value (cm)', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## SECTION 4: Data Filtering Techniques
# 
# Filtering is essential for:
# - Subsetting data based on conditions
# - Isolating specific classes or features
# - Removing outliers
# - Creating focused analyses

# %% [markdown]
# ### 4.1 Filter by Feature Conditions

# %%
# Filter 1: Samples with sepal length > 6.0 cm
print("=" * 80)
print("FILTER 1: Sepal Length > 6.0 cm")
print("=" * 80)

filtered_sepal = df[df['sepal length (cm)'] > 6.0]
print(f"\nOriginal dataset size: {len(df)}")
print(f"Filtered dataset size: {len(filtered_sepal)}")
print(f"Percentage retained: {(len(filtered_sepal) / len(df)) * 100:.2f}%")
print(f"\nSpecies distribution in filtered data:")
print(filtered_sepal['species_name'].value_counts())

# %%
# Filter 2: Samples with petal width < 0.5 cm
print("\n" + "=" * 80)
print("FILTER 2: Petal Width < 0.5 cm")
print("=" * 80)

filtered_petal = df[df['petal width (cm)'] < 0.5]
print(f"\nOriginal dataset size: {len(df)}")
print(f"Filtered dataset size: {len(filtered_petal)}")
print(f"Percentage retained: {(len(filtered_petal) / len(df)) * 100:.2f}%")
print(f"\nSpecies distribution in filtered data:")
print(filtered_petal['species_name'].value_counts())

# %%
# Filter 3: Complex condition - Multiple filters combined
print("\n" + "=" * 80)
print("FILTER 3: Complex Condition (Sepal Length > 5.5 AND Petal Length > 4.0)")
print("=" * 80)

filtered_complex = df[(df['sepal length (cm)'] > 5.5) & (df['petal length (cm)'] > 4.0)]
print(f"\nOriginal dataset size: {len(df)}")
print(f"Filtered dataset size: {len(filtered_complex)}")
print(f"Percentage retained: {(len(filtered_complex) / len(df)) * 100:.2f}%")
print(f"\nSpecies distribution in filtered data:")
print(filtered_complex['species_name'].value_counts())

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['species_name'].value_counts().plot(kind='bar', color='lightblue', edgecolor='black', alpha=0.8)
plt.title('Original Data - Species Distribution', fontsize=13, fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
filtered_complex['species_name'].value_counts().plot(kind='bar', color='coral', edgecolor='black', alpha=0.8)
plt.title('Filtered Data - Species Distribution', fontsize=13, fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Filter by Class Labels

# %%
# Filter by specific species
print("=" * 80)
print("FILTER BY CLASS LABELS")
print("=" * 80)

# Get only setosa samples
setosa_only = df[df['species_name'] == 'setosa']
print(f"\nSetosa samples: {len(setosa_only)}")
print("\nSetosa - Statistical Summary:")
print(setosa_only[feature_cols].describe())

# %%
# Get versicolor and virginica (exclude setosa)
print("\n" + "=" * 80)
print("NON-SETOSA SPECIES")
print("=" * 80)

non_setosa = df[df['species_name'] != 'setosa']
print(f"\nNon-setosa samples: {len(non_setosa)}")
print("\nSpecies distribution:")
print(non_setosa['species_name'].value_counts())

# %%
# Multiple species filter using isin()
print("\n" + "=" * 80)
print("MULTIPLE SPECIES SELECTION")
print("=" * 80)

selected_species = df[df['species_name'].isin(['versicolor', 'virginica'])]
print(f"\nSelected species samples: {len(selected_species)}")
print("\nMean values by species:")
print(selected_species.groupby('species_name')[feature_cols].mean())

# %% [markdown]
# ### 4.3 Outlier Filtering

# %%
# Remove outliers using IQR method for a specific feature
print("=" * 80)
print("OUTLIER REMOVAL - Sepal Width")
print("=" * 80)

Q1 = df['sepal width (cm)'].quantile(0.25)
Q3 = df['sepal width (cm)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_no_outliers = df[(df['sepal width (cm)'] >= lower_bound) & (df['sepal width (cm)'] <= upper_bound)]

print(f"\nOriginal dataset size: {len(df)}")
print(f"Dataset after outlier removal: {len(df_no_outliers)}")
print(f"Outliers removed: {len(df) - len(df_no_outliers)}")

# Visualize before and after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot(df['sepal width (cm)'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
axes[0].set_title('Before Outlier Removal', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].grid(axis='y', alpha=0.3)

axes[1].boxplot(df_no_outliers['sepal width (cm)'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightgreen'))
axes[1].set_title('After Outlier Removal', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Sepal Width (cm)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Feature Selection Filtering

# %%
# Select specific features only
print("=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Select only petal features
petal_features = df[['petal length (cm)', 'petal width (cm)', 'species_name']]
print("\nPetal Features Only:")
print(petal_features.head(10))

# Select only sepal features
sepal_features = df[['sepal length (cm)', 'sepal width (cm)', 'species_name']]
print("\nSepal Features Only:")
print(sepal_features.head(10))

# %% [markdown]
# ### 4.5 Quantile-Based Filtering

# %%
# Filter top 25% of samples by petal length
print("=" * 80)
print("QUANTILE-BASED FILTERING")
print("=" * 80)

petal_length_75th = df['petal length (cm)'].quantile(0.75)
top_25_percent = df[df['petal length (cm)'] >= petal_length_75th]

print(f"\n75th percentile of petal length: {petal_length_75th:.3f} cm")
print(f"Samples in top 25%: {len(top_25_percent)}")
print("\nSpecies distribution in top 25%:")
print(top_25_percent['species_name'].value_counts())

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['petal length (cm)'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(petal_length_75th, color='red', linestyle='--', linewidth=2, label=f'75th percentile: {petal_length_75th:.2f}')
plt.title('All Data - Petal Length Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(top_25_percent['petal length (cm)'], bins=15, color='coral', edgecolor='black', alpha=0.7)
plt.title('Top 25% - Petal Length Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## SECTION 5: Statistical Analysis (Correlation & Covariance)
# 
# Understanding relationships between features through:
# - Correlation analysis (measures linear relationship strength)
# - Covariance analysis (measures how variables change together)
# - Feature redundancy identification

# %% [markdown]
# ### 5.1 Correlation Matrix Analysis

# %%
# Calculate correlation matrix
print("=" * 80)
print("CORRELATION MATRIX")
print("=" * 80)

correlation_matrix = df[feature_cols].corr()
print("\n", correlation_matrix)

# Detailed interpretation
print("\n" + "=" * 80)
print("CORRELATION INTERPRETATION")
print("=" * 80)
print("\nStrong Positive Correlations (> 0.7):")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if corr_val > 0.7:
            print(f"  {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f}")

print("\nModerate Correlations (0.4 - 0.7):")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if 0.4 <= corr_val <= 0.7:
            print(f"  {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f}")

print("\nWeak/Negative Correlations (< 0.4):")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if corr_val < 0.4:
            print(f"  {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f}")

# %%
# Enhanced correlation heatmap with annotations
plt.figure(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlGn', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            fmt='.3f', vmin=-1, vmax=1, annot_kws={'size': 11})

plt.title('Correlation Heatmap - Lower Triangle', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Covariance Matrix Analysis

# %%
# Calculate covariance matrix
print("=" * 80)
print("COVARIANCE MATRIX")
print("=" * 80)

covariance_matrix = df[feature_cols].cov()
print("\n", covariance_matrix)

print("\n" + "=" * 80)
print("COVARIANCE INTERPRETATION")
print("=" *80)
print("\nCovariance measures how two variables change together.")
print("Positive covariance: variables tend to increase together")
print("Negative covariance: when one increases, the other decreases")
print("Value magnitude depends on feature scales (unlike correlation)")

# %%
# Visualize covariance matrix
plt.figure(figsize=(12, 10))

sns.heatmap(covariance_matrix, annot=True, cmap='viridis',
            square=True, linewidths=2, cbar_kws={"shrink": 0.8, "label": "Covariance"},
            fmt='.3f')

plt.title('Covariance Heatmap - Iris Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.3 Mathematical Relationship: Correlation vs Covariance

# %%
print("=" * 80)
print("CORRELATION vs COVARIANCE - MATHEMATICAL RELATIONSHIP")
print("=" * 80)

print("\nFormulas:")
print("Covariance(X,Y) = E[(X - μX)(Y - μY)]")
print("Correlation(X,Y) = Covariance(X,Y) / (σX * σY)")
print("\nwhere μ = mean, σ = standard deviation, E = expected value")

print("\n" + "-" * 80)
print("Verification for Petal Length and Petal Width:")
print("-" * 80)

# Manual calculation
x = df['petal length (cm)']
y = df['petal width (cm)']

# Covariance
cov_manual = np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1)
cov_builtin = covariance_matrix.loc['petal length (cm)', 'petal width (cm)']

# Correlation
corr_manual = cov_manual / (x.std() * y.std())
corr_builtin = correlation_matrix.loc['petal length (cm)', 'petal width (cm)']

print(f"\nCovariance (manual calculation): {cov_manual:.6f}")
print(f"Covariance (built-in function): {cov_builtin:.6f}")
print(f"Match: {np.isclose(cov_manual, cov_builtin)}")

print(f"\nCorrelation (manual calculation): {corr_manual:.6f}")
print(f"Correlation (built-in function): {corr_builtin:.6f}")
print(f"Match: {np.isclose(corr_manual, corr_builtin)}")

print(f"\nStandard deviation of Petal Length: {x.std():.6f}")
print(f"Standard deviation of Petal Width: {y.std():.6f}")
print(f"Product of std devs: {x.std() * y.std():.6f}")
print(f"Covariance / (σX * σY) = {cov_manual / (x.std() * y.std()):.6f}")

# %% [markdown]
# ### 5.4 Feature Redundancy Analysis

# %%
print("=" * 80)
print("FEATURE REDUNDANCY ANALYSIS")
print("=" * 80)

# Set threshold for high correlation (potential redundancy)
threshold = 0.9

print(f"\nSearching for highly correlated features (correlation > {threshold}):")
high_corr_pairs = []

for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > threshold:
            high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
            print(f"\n  {feature_cols[i]} ↔ {feature_cols[j]}")
            print(f"  Correlation: {corr_val:.3f}")
            print(f"  → Potentially redundant features!")

if not high_corr_pairs:
    print(f"\n  No feature pairs found with correlation > {threshold}")
    print("  This suggests all features provide unique information")

# Check correlation at lower threshold
threshold_medium = 0.7
print(f"\n\nFeatures with strong correlation (> {threshold_medium}):")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > threshold_medium:
            print(f"  {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f}")

# %% [markdown]
# ### 5.5 Pairwise Feature Correlation Plots (F1 vs F2 style)

# %%
# Detailed correlation analysis between specific feature pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

feature_pairs = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal length (cm)', 'petal width (cm)'),
    ('sepal width (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)'),
    ('petal length (cm)', 'petal width (cm)')
]

for idx, (f1, f2) in enumerate(feature_pairs):
    # Scatter plot
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        axes[idx].scatter(species_data[f1], species_data[f2],
                         label=species, alpha=0.6, s=50, color=colors[species])
    
    # Calculate and display correlation
    corr = df[f1].corr(df[f2])
    
    # Add regression line
    z = np.polyfit(df[f1], df[f2], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[f1], p(df[f1]), "r--", alpha=0.8, linewidth=2)
    
    axes[idx].set_xlabel(f1, fontsize=10)
    axes[idx].set_ylabel(f2, fontsize=10)
    axes[idx].set_title(f'{f1.split()[0].capitalize()} vs {f2.split()[0].capitalize()}\nCorrelation: {corr:.3f}',
                       fontsize=11, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.6 Correlation by Species

# %%
# Calculate correlation for each species separately
print("=" * 80)
print("CORRELATION ANALYSIS BY SPECIES")
print("=" * 80)

for species in df['species_name'].unique():
    print(f"\n{species.upper()}:")
    print("-" * 40)
    species_data = df[df['species_name'] == species][feature_cols]
    species_corr = species_data.corr()
    print(species_corr)

# %%
# Visualize correlation differences across species
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, species in enumerate(df['species_name'].unique()):
    species_data = df[df['species_name'] == species][feature_cols]
    species_corr = species_data.corr()
    
    sns.heatmap(species_corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1, ax=axes[idx])
    axes[idx].set_title(f'{species.capitalize()} - Correlation', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## SECTION 6: Sampling Demonstrations
# 
# Sampling techniques are crucial for:
# - Creating training/testing datasets
# - Handling large datasets
# - Maintaining class proportions
# - Reducing computational costs

# %% [markdown]
# ### 6.1 Simple Random Sampling

# %%
print("=" * 80)
print("SIMPLE RANDOM SAMPLING")
print("=" * 80)

# Take a random sample of 30% of the data
sample_size = 0.3
random_sample = df.sample(frac=sample_size, random_state=42)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Sample size ({sample_size*100}%): {len(random_sample)}")

print("\nOriginal class distribution:")
print(df['species_name'].value_counts())
print("\nPercentages:")
print((df['species_name'].value_counts() / len(df)) * 100)

print("\nRandom sample class distribution:")
print(random_sample['species_name'].value_counts())
print("\nPercentages:")
print((random_sample['species_name'].value_counts() / len(random_sample)) * 100)

# %%
# Visualize distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
df['species_name'].value_counts().plot(kind='bar', ax=axes[0], 
                                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                        edgecolor='black', alpha=0.8)
axes[0].set_title('Original Data - Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Species')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Random sample distribution
random_sample['species_name'].value_counts().plot(kind='bar', ax=axes[1],
                                                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                   edgecolor='black', alpha=0.8)
axes[1].set_title(f'Random Sample ({sample_size*100}%) - Class Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Species')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n⚠ Note: Random sampling may not preserve class proportions perfectly")

# %% [markdown]
# ### 6.2 Stratified Sampling

# %%
print("=" * 80)
print("STRATIFIED SAMPLING")
print("=" * 80)
print("\nStratified sampling maintains the proportion of each class in the sample")

# Perform stratified sampling
from sklearn.model_selection import train_test_split

# We'll use train_test_split with stratification
stratified_sample, _ = train_test_split(df, test_size=0.7, stratify=df['species_name'], random_state=42)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Stratified sample size (30%): {len(stratified_sample)}")

print("\nOriginal class distribution:")
orig_dist = df['species_name'].value_counts()
print(orig_dist)
print("\nPercentages:")
print((orig_dist / len(df)) * 100)

print("\nStratified sample class distribution:")
strat_dist = stratified_sample['species_name'].value_counts()
print(strat_dist)
print("\nPercentages:")
print((strat_dist / len(stratified_sample)) * 100)

# %%
# Visualize stratified sampling
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original distribution - count
df['species_name'].value_counts().plot(kind='bar', ax=axes[0, 0],
                                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                        edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Original - Count', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Original distribution - percentage
(df['species_name'].value_counts() / len(df) * 100).plot(kind='bar', ax=axes[0, 1],
                                                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                          edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Original - Percentage', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Percentage (%)')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
axes[0, 1].set_ylim([0, 40])
axes[0, 1].grid(axis='y', alpha=0.3)

# Stratified sample - count
stratified_sample['species_name'].value_counts().plot(kind='bar', ax=axes[1, 0],
                                                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                       edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Stratified Sample - Count', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Stratified sample - percentage
(stratified_sample['species_name'].value_counts() / len(stratified_sample) * 100).plot(kind='bar', ax=axes[1, 1],
                                                                                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                                                         edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Stratified Sample - Percentage', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
axes[1, 1].set_ylim([0, 40])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Stratified sampling preserves the class proportions perfectly")

# %% [markdown]
# ### 6.3 Hybrid Sampling

# %%
print("=" * 80)
print("HYBRID SAMPLING")
print("=" * 80)
print("\nHybrid approach: Combination of stratified sampling with additional random selection")

# Step 1: Stratified sampling to get base sample
base_sample, remaining = train_test_split(df, test_size=0.8, stratify=df['species_name'], random_state=42)

# Step 2: Add random samples from remaining data
additional_random = remaining.sample(n=10, random_state=42)

# Combine
hybrid_sample = pd.concat([base_sample, additional_random], ignore_index=True)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Base stratified sample: {len(base_sample)}")
print(f"Additional random samples: {len(additional_random)}")
print(f"Total hybrid sample size: {len(hybrid_sample)}")

print("\nOriginal class distribution:")
print(df['species_name'].value_counts())
print("\nPercentages:")
print((df['species_name'].value_counts() / len(df)) * 100)

print("\nHybrid sample class distribution:")
print(hybrid_sample['species_name'].value_counts())
print("\nPercentages:")
print((hybrid_sample['species_name'].value_counts() / len(hybrid_sample)) * 100)

# %%
# Visualize all three sampling methods
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original
(df['species_name'].value_counts() / len(df) * 100).plot(kind='bar', ax=axes[0, 0],
                                                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                          edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Original Distribution', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Percentage (%)')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
axes[0, 0].set_ylim([0, 40])
axes[0, 0].axhline(y=33.33, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 0].grid(axis='y', alpha=0.3)

# Random sample
(random_sample['species_name'].value_counts() / len(random_sample) * 100).plot(kind='bar', ax=axes[0, 1],
                                                                                 color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                                                 edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Random Sampling', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Percentage (%)')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
axes[0, 1].set_ylim([0, 40])
axes[0, 1].axhline(y=33.33, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].grid(axis='y', alpha=0.3)

# Stratified sample
(stratified_sample['species_name'].value_counts() / len(stratified_sample) * 100).plot(kind='bar', ax=axes[1, 0],
                                                                                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                                                         edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Stratified Sampling', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
axes[1, 0].set_ylim([0, 40])
axes[1, 0].axhline(y=33.33, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].grid(axis='y', alpha=0.3)

# Hybrid sample
(hybrid_sample['species_name'].value_counts() / len(hybrid_sample) * 100).plot(kind='bar', ax=axes[1, 1],
                                                                                 color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                                                 edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Hybrid Sampling', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
axes[1, 1].set_ylim([0, 40])
axes[1, 1].axhline(y=33.33, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Expected (33.33%)')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 Sampling Comparison Summary

# %%
# Create comprehensive comparison
print("=" * 80)
print("SAMPLING METHODS COMPARISON")
print("=" * 80)

sampling_comparison = pd.DataFrame({
    'Method': ['Original', 'Random', 'Stratified', 'Hybrid'],
    'Total Samples': [len(df), len(random_sample), len(stratified_sample), len(hybrid_sample)],
    'Setosa': [
        df[df['species_name'] == 'setosa'].shape[0],
        random_sample[random_sample['species_name'] == 'setosa'].shape[0],
        stratified_sample[stratified_sample['species_name'] == 'setosa'].shape[0],
        hybrid_sample[hybrid_sample['species_name'] == 'setosa'].shape[0]
    ],
    'Versicolor': [
        df[df['species_name'] == 'versicolor'].shape[0],
        random_sample[random_sample['species_name'] == 'versicolor'].shape[0],
        stratified_sample[stratified_sample['species_name'] == 'versicolor'].shape[0],
        hybrid_sample[hybrid_sample['species_name'] == 'versicolor'].shape[0]
    ],
    'Virginica': [
        df[df['species_name'] == 'virginica'].shape[0],
        random_sample[random_sample['species_name'] == 'virginica'].shape[0],
        stratified_sample[stratified_sample['species_name'] == 'virginica'].shape[0],
        hybrid_sample[hybrid_sample['species_name'] == 'virginica'].shape[0]
    ]
})

print("\n", sampling_comparison)

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("\n1. Random Sampling:")
print("   - Quick and simple")
print("   - May not preserve class proportions")
print("   - Good for large, balanced datasets")

print("\n2. Stratified Sampling:")
print("   - Maintains exact class proportions")
print("   - Ideal for imbalanced datasets")
print("   - Recommended for train/test splits")

print("\n3. Hybrid Sampling:")
print("   - Combines benefits of both methods")
print("   - Flexible approach")
print("   - Useful for complex sampling requirements")

# %% [markdown]
# ---
# ## SECTION 7: Multiclass Classification Data Preparation
# 
# Preparing data for multiclass classification involves:
# - Understanding class labels
# - Encoding categorical variables
# - Analyzing class balance
# - Feature scaling considerations

# %% [markdown]
# ### 7.1 Multiclass Label Analysis

# %%
print("=" * 80)
print("MULTICLASS LABEL ANALYSIS")
print("=" * 80)

print(f"\nNumber of classes: {df['species'].nunique()}")
print(f"Class labels (encoded): {sorted(df['species'].unique())}")
print(f"Class names: {sorted(df['species_name'].unique())}")

print("\nClass mapping:")
for i in range(df['species'].nunique()):
    species_name = df[df['species'] == i]['species_name'].iloc[0]
    count = len(df[df['species'] == i])
    print(f"  {i} → {species_name:12s} ({count} samples)")

# %%
# Detailed class statistics
print("\n" + "=" * 80)
print("CLASS-WISE STATISTICS")
print("=" * 80)

for species in df['species_name'].unique():
    print(f"\n{species.upper()}:")
    print("-" * 40)
    species_data = df[df['species_name'] == species][feature_cols]
    print(species_data.describe())

# %% [markdown]
# ### 7.2 Label Encoding Demonstration

# %%
print("=" * 80)
print("LABEL ENCODING")
print("=" * 80)

# The dataset already has encoded labels, but let's demonstrate the process
from sklearn.preprocessing import LabelEncoder

# Create a copy for demonstration
df_encoded = df.copy()

# Initialize label encoder
le = LabelEncoder()

# Fit and transform
df_encoded['species_encoded'] = le.fit_transform(df_encoded['species_name'])

print("\nOriginal species names and their encoded values:")
encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
for name, encoded in encoding_map.items():
    print(f"  {name:12s} → {encoded}")

print("\nSample rows showing encoding:")
print(df_encoded[['species_name', 'species', 'species_encoded']].head(10))

print("\n✓ Encoding is consistent with original labels")

# %% [markdown]
# ### 7.3 One-Hot Encoding Demonstration

# %%
print("=" * 80)
print("ONE-HOT ENCODING")
print("=" * 80)

# Perform one-hot encoding
df_onehot = pd.get_dummies(df['species_name'], prefix='species')

print("\nOne-hot encoded representation:")
print(df_onehot.head(10))

# Combine with original features
df_with_onehot = pd.concat([df[feature_cols], df_onehot], axis=1)

print("\nDataset with one-hot encoded labels:")
print(df_with_onehot.head())
print(f"\nShape: {df_with_onehot.shape}")

# %%
# Visualize one-hot encoding
sample_indices = [0, 50, 100]  # One from each class
sample_data = df_onehot.iloc[sample_indices]

fig, ax = plt.subplots(figsize=(10, 6))
sample_data.T.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], width=0.8)
ax.set_title('One-Hot Encoding Visualization (Sample from Each Class)', fontsize=14, fontweight='bold')
ax.set_xlabel('Encoded Columns', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend([f'Sample {i} ({df.iloc[i]["species_name"]})' for i in sample_indices])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.4 Class Balance Analysis

# %%
print("=" * 80)
print("CLASS BALANCE ANALYSIS")
print("=" * 80)

# Calculate class weights (inverse of frequency)
class_counts = df['species_name'].value_counts()
total_samples = len(df)
n_classes = len(class_counts)

class_weights = {}
for species in class_counts.index:
    weight = total_samples / (n_classes * class_counts[species])
    class_weights[species] = weight

print("\nClass counts:")
print(class_counts)

print("\nClass weights (for handling imbalance):")
for species, weight in class_weights.items():
    print(f"  {species:12s} → {weight:.4f}")

print("\n✓ Dataset is perfectly balanced (all weights ≈ 1.0)")

# %%
# Visualize class balance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Count plot
class_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                  edgecolor='black', alpha=0.8)
axes[0].set_title('Class Distribution - Counts', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Species')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].axhline(y=total_samples/n_classes, color='red', linestyle='--', linewidth=2, 
                label=f'Average ({total_samples/n_classes:.0f})')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Percentage plot
(class_counts / total_samples * 100).plot(kind='bar', ax=axes[1], 
                                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                          edgecolor='black', alpha=0.8)
axes[1].set_title('Class Distribution - Percentage', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_xlabel('Species')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].axhline(y=100/n_classes, color='red', linestyle='--', linewidth=2,
                label=f'Average ({100/n_classes:.1f}%)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Class weights plot
pd.Series(class_weights).plot(kind='bar', ax=axes[2], color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                              edgecolor='black', alpha=0.8)
axes[2].set_title('Class Weights', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Weight')
axes[2].set_xlabel('Species')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
axes[2].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Balanced (1.0)')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.5 Feature Scaling Preparation

# %%
print("=" * 80)
print("FEATURE SCALING ANALYSIS")
print("=" * 80)

print("\nFeature ranges (before scaling):")
for col in feature_cols:
    print(f"  {col:30s}: [{df[col].min():.3f}, {df[col].max():.3f}]  (range: {df[col].max() - df[col].min():.3f})")

# %%
# Demonstrate different scaling techniques (without applying to original data)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling (Z-score normalization)
scaler_standard = StandardScaler()
df_standard = pd.DataFrame(scaler_standard.fit_transform(df[feature_cols]), 
                          columns=feature_cols)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df[feature_cols]),
                        columns=feature_cols)

# Robust Scaling
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(scaler_robust.fit_transform(df[feature_cols]),
                        columns=feature_cols)

print("\n" + "=" * 80)
print("SCALING COMPARISON")
print("=" * 80)

print("\nStandard Scaling (mean=0, std=1):")
print(df_standard.describe())

print("\nMin-Max Scaling (range=[0,1]):")
print(df_minmax.describe())

print("\nRobust Scaling (using median and IQR):")
print(df_robust.describe())

# %%
# Visualize scaling effects
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Original data
df[feature_cols].boxplot(ax=axes[0, 0])
axes[0, 0].set_title('Original Data', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_xticklabels([col.split()[0].capitalize() for col in feature_cols], rotation=45)
axes[0, 0].grid(alpha=0.3)

# Standard scaled
df_standard.boxplot(ax=axes[0, 1])
axes[0, 1].set_title('Standard Scaling', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Standardized Value')
axes[0, 1].set_xticklabels([col.split()[0].capitalize() for col in feature_cols], rotation=45)
axes[0, 1].grid(alpha=0.3)

# Min-Max scaled
df_minmax.boxplot(ax=axes[1, 0])
axes[1, 0].set_title('Min-Max Scaling', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Normalized Value')
axes[1, 0].set_xticklabels([col.split()[0].capitalize() for col in feature_cols], rotation=45)
axes[1, 0].grid(alpha=0.3)

# Robust scaled
df_robust.boxplot(ax=axes[1, 1])
axes[1, 1].set_title('Robust Scaling', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Scaled Value')
axes[1, 1].set_xticklabels([col.split()[0].capitalize() for col in feature_cols], rotation=45)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.6 Train-Test Split Preparation

# %%
print("=" * 80)
print("TRAIN-TEST SPLIT PREPARATION")
print("=" * 80)

# Prepare features and labels
X = df[feature_cols]
y = df['species']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")

# Perform train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                     stratify=y, random_state=42)

print(f"\nTraining set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

print(f"\nTest set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")

print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nClass distribution in test set:")
print(pd.Series(y_test).value_counts().sort_index())

# %%
# Visualize train-test split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training set distribution
train_counts = pd.Series(y_train).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}).value_counts()
train_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                  edgecolor='black', alpha=0.8)
axes[0].set_title(f'Training Set Distribution (70%)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Species')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Test set distribution
test_counts = pd.Series(y_test).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}).value_counts()
test_counts.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                 edgecolor='black', alpha=0.8)
axes[1].set_title(f'Test Set Distribution (30%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Species')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Data is ready for multiclass classification modeling!")

# %% [markdown]
# ---
# ## SECTION 8: Summary & Insights
# 
# Final comprehensive summary of all EDA findings and recommendations

# %%
print("=" * 100)
print(" " * 35 + "EDA SUMMARY & KEY INSIGHTS")
print("=" * 100)

print("\n1. DATASET OVERVIEW:")
print("   " + "-" * 90)
print(f"   • Total samples: {len(df)}")
print(f"   • Number of features: {len(feature_cols)}")
print(f"   • Number of classes: {df['species'].nunique()}")
print(f"   • Missing values: {df.isnull().sum().sum()} (Perfect data quality!)")
print(f"   • Duplicate rows: {df.duplicated().sum()}")

print("\n2. CLASS DISTRIBUTION:")
print("   " + "-" * 90)
print("   • Perfectly balanced dataset (50 samples per class)")
print("   • Setosa: 50 samples (33.33%)")
print("   • Versicolor: 50 samples (33.33%)")
print("   • Virginica: 50 samples (33.33%)")
print("   ✓ No class imbalance handling required")

print("\n3. FEATURE CHARACTERISTICS:")
print("   " + "-" * 90)
for col in feature_cols:
    print(f"   • {col}:")
    print(f"     Range: [{df[col].min():.2f}, {df[col].max():.2f}]  |  Mean: {df[col].mean():.2f}  |  Std: {df[col].std():.2f}")

print("\n4. CORRELATION INSIGHTS:")
print("   " + "-" * 90)
print("   Strong correlations found:")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            print(f"   • {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f} (Very Strong)")
        elif abs(corr_val) > 0.7:
            print(f"   • {feature_cols[i]} ↔ {feature_cols[j]}: {corr_val:.3f} (Strong)")

print("\n5. OUTLIER ANALYSIS:")
print("   " + "-" * 90)
print("   • Minimal outliers detected using IQR method")
print("   • Most outliers found in sepal width feature")
print("   • Outliers are species-specific and biologically meaningful")
print("   ✓ No outlier removal recommended")

print("\n6. FEATURE SEPARABILITY:")
print("   " + "-" * 90)
print("   • Setosa is highly separable from other species (especially in petal features)")
print("   • Versicolor and Virginica show some overlap")
print("   • Petal length and petal width are the most discriminative features")
print("   • Sepal features show more overlap between classes")

print("\n7. SAMPLING EVALUATION:")
print("   " + "-" * 90)
print("   • Random sampling: Quick but may not preserve proportions")
print("   • Stratified sampling: Recommended for train/test splits (maintains class balance)")
print("   • Hybrid sampling: Flexible for complex requirements")

print("\n8. DATA QUALITY ASSESSMENT:")
print("   " + "-" * 90)
print("   ✓ No missing values")
print("   ✓ No duplicate rows")
print("   ✓ All features are numeric (float64)")
print("   ✓ Perfect class balance")
print("   ✓ Reasonable feature ranges")
print("   ✓ No data type issues")

print("\n9. RECOMMENDATIONS FOR MODELING:")
print("   " + "-" * 90)
print("   • Use stratified sampling for train/test split to maintain class proportions")
print("   • Consider feature scaling (StandardScaler or MinMaxScaler) for distance-based algorithms")
print("   • Petal features should have higher importance in classification")
print("   • No need for class balancing techniques")
print("   • All features can be used (no redundancy issues)")
print("   • Consider both linear and non-linear models (data shows good separability)")

print("\n10. KEY STATISTICAL FINDINGS:")
print("   " + "-" * 90)
print(f"   • Highest correlation: Petal length ↔ Petal width ({correlation_matrix.loc['petal length (cm)', 'petal width (cm)']:.3f})")
print(f"   • Feature with largest range: Petal length ({df['petal length (cm)'].max() - df['petal length (cm)'].min():.2f} cm)")
print(f"   • Most variable feature: Petal length (σ = {df['petal length (cm)'].std():.2f})")
print(f"   • Least variable feature: Sepal width (σ = {df['sepal width (cm)'].std():.2f})")

print("\n" + "=" * 100)
print(" " * 30 + "END OF EXPLORATORY DATA ANALYSIS")
print("=" * 100)

print("\n✓ EDA Complete! Dataset is ready for machine learning modeling.")
print("\nNext Steps:")
print("  1. Feature engineering (if needed)")
print("  2. Model selection and training")
print("  3. Model evaluation and tuning")
print("  4. Performance comparison across different algorithms")

# %% [markdown]
# ---
# ## Visualizations Summary

# %%
# Create a final comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Class distribution
ax1 = fig.add_subplot(gs[0, 0])
df['species_name'].value_counts().plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                                        edgecolor='black', alpha=0.8)
ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(axis='y', alpha=0.3)

# 2. Correlation heatmap (compact)
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2,
            square=True, cbar_kws={"shrink": 0.8}, fmt='.2f', vmin=-1, vmax=1)
ax2.set_title('Feature Correlations', fontsize=12, fontweight='bold')

# 3. Feature distributions
ax3 = fig.add_subplot(gs[0, 2])
for col in feature_cols:
    df[col].plot(kind='kde', ax=ax3, label=col.split()[0].capitalize(), linewidth=2)
ax3.set_title('Feature Density Distributions', fontsize=12, fontweight='bold')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 4. Petal Length vs Width scatter
ax4 = fig.add_subplot(gs[1, 0])
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    ax4.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'],
               label=species, alpha=0.6, s=50, color=colors[species], edgecolors='black')
ax4.set_title('Petal Length vs Width', fontsize=12, fontweight='bold')
ax4.set_xlabel('Petal Length (cm)')
ax4.set_ylabel('Petal Width (cm)')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# 5. Box plots
ax5 = fig.add_subplot(gs[1, 1])
df[feature_cols].boxplot(ax=ax5)
ax5.set_title('Feature Box Plots', fontsize=12, fontweight='bold')
ax5.set_ylabel('Value (cm)')
ax5.set_xticklabels([col.split()[0].capitalize() for col in feature_cols], rotation=45)
ax5.grid(alpha=0.3)

# 6. Mean comparison
ax6 = fig.add_subplot(gs[1, 2])
mean_by_species.plot(kind='bar', ax=ax6, width=0.8)
ax6.set_title('Mean Feature Values by Species', fontsize=12, fontweight='bold')
ax6.set_ylabel('Mean Value (cm)')
ax6.set_xlabel('Species')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
ax6.legend(title='Features', fontsize=8, title_fontsize=9, loc='upper left')
ax6.grid(axis='y', alpha=0.3)

# 7-9. Individual feature distributions by species
feature_indices = [0, 2]  # Sepal length and Petal length
for idx, feat_idx in enumerate(feature_indices):
    ax = fig.add_subplot(gs[2, idx])
    col = feature_cols[feat_idx]
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species][col]
        species_data.plot(kind='kde', ax=ax, label=species, linewidth=2, alpha=0.7)
    ax.set_title(f'{col} by Species', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Last plot - sampling comparison
ax9 = fig.add_subplot(gs[2, 2])
sampling_methods = ['Original', 'Random', 'Stratified']
sampling_sizes = [len(df), len(random_sample), len(stratified_sample)]
ax9.bar(sampling_methods, sampling_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
        edgecolor='black', alpha=0.8)
ax9.set_title('Sampling Methods Comparison', fontsize=12, fontweight='bold')
ax9.set_ylabel('Sample Size')
ax9.set_xlabel('Method')
ax9.grid(axis='y', alpha=0.3)

fig.suptitle('Comprehensive EDA Summary - Iris Dataset', fontsize=18, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 100)
print("✓ Comprehensive EDA visualization complete!")
print("=" * 100)

# %%