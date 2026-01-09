# %% [markdown]
# # Section A: Structural and Contextual Exploration
# ## Video Game Sales Dataset Analysis

# %% [markdown]
# ### Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## 5.1 Data Context and Generation
# 
# ### Understanding the Dataset Origin
# 
# **What real-world process produced this data?**
# - This dataset represents video game sales records scraped from VGChartz, a video game sales tracking website
# - Data aggregates retail sales across different geographic regions
# - Collection likely spans multiple decades of gaming industry history
# 
# **Who or what is represented by each row?**
# - Each row represents a single video game title release
# - Observations capture game sales performance across regions and platforms
# 
# **What does one observation truly mean?**
# - One observation = one game's commercial performance metrics
# - Includes platform, publisher, genre, release year, and regional sales figures
# 
# **What factors might influence data quality?**
# - VGChartz relies on estimation algorithms for some sales figures
# - Digital sales may be underrepresented (dataset focuses on physical copies)
# - Regional reporting inconsistencies
# - Missing data for older or less popular titles
# - Potential bias toward Western markets

# %% [markdown]
# ### Assumptions Regarding Data Origin
# 
# **Explicitly Stated Assumptions:**
# 1. Sales figures are primarily retail/physical copies
# 2. Data collection methodology remained consistent across years
# 3. Regional sales categories are mutually exclusive
# 4. Publisher and platform names are standardized
# 5. Year represents initial release date, not re-releases

# %% [markdown]
# ## 5.2 Data Structure and Integrity

# %% [markdown]
# ### Loading and Initial Inspection

# %%
# Load the dataset
df = pd.read_csv('vgsales.csv')

# Display first few rows
print("First 5 rows of the dataset:")
df.head()

# %% [markdown]
# ### Dataset Dimensions and Schema

# %%
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nTotal observations: {df.shape[0]:,}")
print(f"Total variables: {df.shape[1]}")

# %%
print("\nDataset Schema:")
print("="*60)
df.info()

# %%
print("\nColumn Names and Types:")
print("="*60)
for col in df.columns:
    print(f"{col:15} | {str(df[col].dtype):10} | Non-null: {df[col].notna().sum():,}")

# %% [markdown]
# ### Data Type Validation and Correction

# %%
# Examine data types in detail
print("\nData Type Summary:")
print("="*60)
print(df.dtypes)

# %%
# Check for type inconsistencies
print("\nSample values for each column:")
print("="*60)
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Sample values: {df[col].dropna().unique()[:5]}")
    print(f"  Data type: {df[col].dtype}")

# %%
# Validate Year column
print("\nYear Column Analysis:")
print(f"  Data type: {df['Year'].dtype}")
print(f"  Unique values: {df['Year'].nunique()}")
print(f"  Range: {df['Year'].min()} to {df['Year'].max()}")
print(f"  Contains NaN: {df['Year'].isna().sum()}")

# Note: Year should ideally be integer, check if conversion needed
if df['Year'].dtype == 'float64':
    print("\n⚠️  Year is float64, likely due to missing values")

# %% [markdown]
# ### Missing Value Analysis

# %%
# Overall missing value count
print("\nMissing Values Summary:")
print("="*60)
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_summary.to_string(index=False))

# %%
# Visualize missing values
plt.figure(figsize=(10, 6))
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    plt.barh(missing_data.index, missing_data.values, color='salmon')
    plt.xlabel('Number of Missing Values')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values detected in the dataset")

# %% [markdown]
# ### Analytical Question: Are missing values random or systematic?

# %%
# Analyze missing value patterns
print("\nMissing Value Pattern Analysis:")
print("="*60)

# Check if Year is the primary source of missing values
if 'Year' in missing_summary['Column'].values:
    year_missing = df[df['Year'].isna()]
    print(f"\nGames with missing Year: {len(year_missing)}")
    print("\nSample of games with missing years:")
    print(year_missing[['Name', 'Platform', 'Publisher', 'Genre']].head(10))
    
    # Check if missing years correlate with other patterns
    print("\nPlatform distribution for missing years:")
    print(year_missing['Platform'].value_counts().head())
    
    print("\nPublisher distribution for missing years:")
    print(year_missing['Publisher'].value_counts().head())

# %%
# Check for Publisher missing values
if 'Publisher' in df.columns:
    publisher_missing = df[df['Publisher'].isna()]
    if len(publisher_missing) > 0:
        print(f"\nGames with missing Publisher: {len(publisher_missing)}")
        print(publisher_missing[['Name', 'Platform', 'Year', 'Genre']].head())

# %% [markdown]
# ### Duplicate and Inconsistent Record Detection

# %%
# Check for duplicate rows
print("\nDuplicate Analysis:")
print("="*60)
duplicates = df.duplicated()
print(f"Total duplicate rows: {duplicates.sum()}")

if duplicates.sum() > 0:
    print("\nSample duplicate rows:")
    print(df[duplicates].head())

# %%
# Check for duplicate game names (potential inconsistencies)
print("\nGame Name Uniqueness:")
print(f"Total unique game names: {df['Name'].nunique()}")
print(f"Total records: {len(df)}")
print(f"Games appearing multiple times: {len(df) - df['Name'].nunique()}")

# %%
# Identify games with multiple entries
name_counts = df['Name'].value_counts()
multiple_entries = name_counts[name_counts > 1]

print(f"\nGames with multiple platform releases: {len(multiple_entries)}")
print("\nTop 10 games by number of platform releases:")
print(multiple_entries.head(10))

# %%
# Example: Examine one game across platforms
if len(multiple_entries) > 0:
    example_game = multiple_entries.index[0]
    print(f"\nExample: '{example_game}' across platforms:")
    print(df[df['Name'] == example_game][['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']])

# %% [markdown]
# ### Analytical Questions: Variable Reliability and Analysis Suitability

# %%
print("\nVariable Reliability Assessment:")
print("="*60)

# Assess each variable
reliability_assessment = {
    'Rank': 'Reliable - Sequential identifier',
    'Name': 'Reliable - Game titles are consistent',
    'Platform': 'Reliable - Standardized platform codes',
    'Year': f'Moderately Reliable - {df["Year"].isna().sum()} missing values ({(df["Year"].isna().sum()/len(df)*100):.1f}%)',
    'Genre': 'Reliable - Categorical with low cardinality',
    'Publisher': f'Moderately Reliable - {df["Publisher"].isna().sum()} missing values' if df["Publisher"].isna().sum() > 0 else 'Reliable',
    'Regional Sales': 'Reliable - Numerical measurements, but estimation-based'
}

for var, assessment in reliability_assessment.items():
    print(f"{var:20} : {assessment}")

# %% [markdown]
# ### Variables That May Distort Results

# %%
print("\nPotential Distortion Analysis:")
print("="*60)

# Check sales columns for scale issues
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

for col in sales_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f} million")
        print(f"  Mean: {df[col].mean():.2f} million")
        print(f"  Median: {df[col].median():.2f} million")
        print(f"  Std Dev: {df[col].std():.2f} million")
        print(f"  Skewness: {df[col].skew():.2f}")

# %%
# Identify potential outliers in Global_Sales
Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR

outliers = df[df['Global_Sales'] > outlier_threshold]
print(f"\nPotential outliers (IQR method): {len(outliers)} games")
print(f"Threshold: {outlier_threshold:.2f} million sales")
print("\nTop sellers (potential scale distortion):")
print(df.nlargest(5, 'Global_Sales')[['Name', 'Platform', 'Year', 'Global_Sales']])

# %% [markdown]
# ## Summary: Section A Findings
# 
# ### Data Context
# - Dataset represents video game sales from VGChartz
# - Each row = one game title release on a specific platform
# - Sales figures likely underrepresent digital distribution
# 
# ### Data Quality
# - **Dimensions**: 16,598 games × 11 variables
# - **Missing Values**: Year (~271 missing), Publisher (minimal)
# - **Missing Pattern**: Appears non-random, concentrated in older/obscure titles
# - **Duplicates**: No exact duplicates, but games appear across multiple platforms
# 
# ### Variable Reliability
# - **Highly Reliable**: Name, Platform, Genre, Sales figures
# - **Moderately Reliable**: Year (1.6% missing), Publisher
# - **Scale Concerns**: Global_Sales highly right-skewed, potential outliers
# 
# ### Analysis Readiness
# - Dataset suitable for exploratory analysis
# - Year missing values require strategy (imputation vs. exclusion)
# - Heavy outliers in sales suggest need for robust statistical methods
# - Multi-platform releases require careful interpretation

# %%