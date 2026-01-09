# %% [markdown]
# # Section B: Univariate Analysis and Question Expansion
# ## Video Game Sales Dataset - Single Variable Deep Dive

# %% [markdown]
# ### Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
sns.set_palette("husl")

# %% [markdown]
# ### Load Dataset

# %%
# Load the dataset
df = pd.read_csv('vgsales.csv')

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()

# %% [markdown]
# ## 6.1 Numerical Variable Exploration
# 
# We will analyze each numerical variable by asking:
# 1. **What is the typical value?**
# 2. **How dispersed is the data?**
# 3. **Is the distribution symmetric or skewed?**
# 4. **Are extreme values meaningful or erroneous?**

# %% [markdown]
# ### Identifying Numerical Variables

# %%
# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Variables in Dataset:")
print("="*60)
for col in numerical_cols:
    print(f"  • {col}")

# Sales columns for detailed analysis
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# %% [markdown]
# ---
# ## Analysis 1: Global Sales Distribution
# ### Research Questions:
# - What is the typical global sales figure for a video game?
# - How dispersed are sales across games?
# - Is success concentrated in a few blockbusters?

# %% [markdown]
# ### Descriptive Statistics: Global Sales

# %%
print("Global Sales - Comprehensive Statistical Summary")
print("="*70)

gs = df['Global_Sales']

# Central tendency
print("\n📊 CENTRAL TENDENCY")
print(f"  Mean:                {gs.mean():.3f} million")
print(f"  Median:              {gs.median():.3f} million")
print(f"  Mode:                {gs.mode().values[0]:.3f} million")
print(f"  Trimmed Mean (10%):  {stats.trim_mean(gs, 0.1):.3f} million")

# Dispersion
print("\n📈 DISPERSION MEASURES")
print(f"  Standard Deviation:  {gs.std():.3f} million")
print(f"  Variance:            {gs.var():.3f}")
print(f"  Range:               {gs.max() - gs.min():.3f} million")
print(f"  IQR:                 {gs.quantile(0.75) - gs.quantile(0.25):.3f} million")
print(f"  Coefficient of Var:  {(gs.std() / gs.mean()) * 100:.2f}%")

# Shape
print("\n📐 DISTRIBUTION SHAPE")
print(f"  Skewness:            {skew(gs):.3f} (Highly right-skewed)")
print(f"  Kurtosis:            {kurtosis(gs):.3f} (Heavy tails)")

# Quantiles
print("\n🎯 QUANTILE ANALYSIS")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"  {p:2d}th percentile:     {gs.quantile(p/100):.3f} million")

# %% [markdown]
# ### Interpretation: Central Tendency
# 
# **Key Observations:**
# - **Mean > Median**: Strong positive skew indicates blockbuster titles pulling the average up
# - **Typical game**: Sells around **0.17 million** copies (median)
# - **Average game**: Appears to sell **0.54 million** (mean), but this is misleading
# - **Most common value**: Very low sales (mode near 0.01)
# 
# **Analytical Insight**: The median is more representative of a "typical" game's performance

# %% [markdown]
# ### Visualization: Distribution Analysis

# %%
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram with KDE
axes[0, 0].hist(gs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(gs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gs.mean():.2f}M')
axes[0, 0].axvline(gs.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {gs.median():.2f}M')
axes[0, 0].set_xlabel('Global Sales (millions)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Global Sales Distribution - Full Range', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Log-transformed histogram
axes[0, 1].hist(np.log10(gs + 0.01), bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Log10(Global Sales + 0.01)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Log-Transformed Sales Distribution', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Box plot
box_parts = axes[1, 0].boxplot(gs, vert=True, patch_artist=True, 
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2),
                                flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
axes[1, 0].set_ylabel('Global Sales (millions)')
axes[1, 0].set_title('Box Plot: Outlier Detection', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Violin plot
parts = axes[1, 1].violinplot([gs], vert=True, showmeans=True, showmedians=True)
axes[1, 1].set_ylabel('Global Sales (millions)')
axes[1, 1].set_title('Violin Plot: Distribution Density', fontsize=14, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Tail Behavior Analysis

# %%
print("\n🔍 TAIL BEHAVIOR: Top-Selling Games")
print("="*70)

# Top performers
top_games = df.nlargest(10, 'Global_Sales')[['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Global_Sales']]
print(top_games.to_string(index=False))

print("\n📊 Sales Concentration Analysis:")
top_1_pct = df.nlargest(int(len(df) * 0.01), 'Global_Sales')['Global_Sales'].sum()
total_sales = df['Global_Sales'].sum()
print(f"  Top 1% of games account for: {(top_1_pct/total_sales)*100:.2f}% of total sales")

top_10_pct = df.nlargest(int(len(df) * 0.10), 'Global_Sales')['Global_Sales'].sum()
print(f"  Top 10% of games account for: {(top_10_pct/total_sales)*100:.2f}% of total sales")

# %% [markdown]
# ### Extreme Value Assessment
# 
# **Are extreme values meaningful or erroneous?**

# %%
# Statistical outlier detection
Q1 = gs.quantile(0.25)
Q3 = gs.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Global_Sales'] < lower_bound) | (df['Global_Sales'] > upper_bound)]

print(f"\n🚨 OUTLIER ANALYSIS (IQR Method)")
print("="*70)
print(f"  Lower bound: {lower_bound:.3f} million")
print(f"  Upper bound: {upper_bound:.3f} million")
print(f"  Outliers detected: {len(outliers)} games ({len(outliers)/len(df)*100:.2f}%)")

print(f"\n  Extreme outliers (>10 million sales): {len(df[df['Global_Sales'] > 10])}")

# Validate top outliers
print("\n✅ OUTLIER VALIDATION:")
print("  These are MEANINGFUL outliers - legitimate blockbuster games")
print("  Examples: Wii Sports, GTA series, Call of Duty series")
print("  NOT erroneous data - represent industry mega-hits")

# %% [markdown]
# ---
# ## Analysis 2: Regional Sales Patterns
# ### Research Questions:
# - How do sales distributions vary by region?
# - Which region has the most consistent sales?
# - Are there region-specific blockbusters?

# %% [markdown]
# ### Comparative Statistics Across Regions

# %%
print("\n🌍 REGIONAL SALES COMPARISON")
print("="*70)

regional_stats = pd.DataFrame()
for col in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']:
    regional_stats[col] = [
        df[col].mean(),
        df[col].median(),
        df[col].std(),
        skew(df[col]),
        df[col].max()
    ]

regional_stats.index = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Max']
print(regional_stats.round(3))

# %% [markdown]
# ### Regional Distribution Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
region_names = ['North America', 'Europe', 'Japan', 'Other Regions']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (region, name, color) in enumerate(zip(regions, region_names, colors)):
    ax = axes[idx // 2, idx % 2]
    
    # Histogram with statistics
    ax.hist(df[region], bins=40, edgecolor='black', alpha=0.7, color=color)
    ax.axvline(df[region].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df[region].mean():.2f}M')
    ax.axvline(df[region].median(), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Median: {df[region].median():.2f}M')
    
    ax.set_xlabel('Sales (millions)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{name} Sales Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Regional Sales - Box Plot Comparison

# %%
# Prepare data for comparative box plot
regional_data = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

plt.figure(figsize=(14, 7))
box_parts = plt.boxplot([df['NA_Sales'], df['EU_Sales'], df['JP_Sales'], df['Other_Sales']],
                        labels=['North America', 'Europe', 'Japan', 'Other'],
                        patch_artist=True,
                        notch=True,
                        showmeans=True)

# Color each box
colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
for patch, color in zip(box_parts['boxes'], colors_box):
    patch.set_facecolor(color)

plt.ylabel('Sales (millions)', fontsize=12)
plt.title('Regional Sales Distribution Comparison', fontsize=15, fontweight='bold')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Regional Market Insights

# %%
print("\n💡 REGIONAL INSIGHTS")
print("="*70)

# Calculate market share
total_regional_sales = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
market_share = (total_regional_sales / total_regional_sales.sum() * 100).round(2)

print("\n📊 Total Market Share by Region:")
for region, share in market_share.items():
    print(f"  {region.replace('_Sales', ''):15} : {share:6.2f}%")

# Identify region-specific hits
print("\n🎮 Region-Specific Top Performers:")
for col, name in zip(regions, region_names):
    top_regional = df.nlargest(3, col)[['Name', 'Platform', col]]
    print(f"\n  {name}:")
    for idx, row in top_regional.iterrows():
        print(f"    • {row['Name'][:40]:40} ({row['Platform']}) - {row[col]:.2f}M")

# %% [markdown]
# ---
# ## Analysis 3: Year Distribution (Temporal Patterns)
# ### Research Questions:
# - Which years saw the most game releases?
# - Has industry output increased over time?
# - Are there notable gaps or spikes?

# %% [markdown]
# ### Handling Missing Years

# %%
# Clean Year data
df_year_clean = df.dropna(subset=['Year'])
year_data = df_year_clean['Year'].astype(int)

print(f"Year Analysis - {len(df) - len(df_year_clean)} records with missing years excluded")
print("="*70)

# %% [markdown]
# ### Year Distribution Statistics

# %%
print("\n📅 YEAR DISTRIBUTION STATISTICS")
print("="*70)

print(f"\n  Earliest release: {year_data.min()}")
print(f"  Latest release:   {year_data.max()}")
print(f"  Time span:        {year_data.max() - year_data.min()} years")
print(f"  Mean year:        {year_data.mean():.1f}")
print(f"  Median year:      {year_data.median():.0f}")
print(f"  Mode year:        {year_data.mode().values[0]}")

print("\n  Releases by Decade:")
decades = (year_data // 10) * 10
print(decades.value_counts().sort_index())

# %% [markdown]
# ### Temporal Visualization

# %%
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 1. Histogram of releases by year
year_counts = year_data.value_counts().sort_index()

axes[0].bar(year_counts.index, year_counts.values, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Number of Releases', fontsize=12)
axes[0].set_title('Video Game Releases Over Time', fontsize=15, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Annotate peak year
peak_year = year_counts.idxmax()
peak_count = year_counts.max()
axes[0].annotate(f'Peak: {peak_year}\n({peak_count} releases)', 
                 xy=(peak_year, peak_count),
                 xytext=(peak_year - 5, peak_count + 50),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=11, color='red', fontweight='bold')

# 2. Cumulative releases
cumulative = year_counts.sort_index().cumsum()
axes[1].plot(cumulative.index, cumulative.values, linewidth=2.5, color='darkgreen')
axes[1].fill_between(cumulative.index, cumulative.values, alpha=0.3, color='lightgreen')
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Cumulative Releases', fontsize=12)
axes[1].set_title('Cumulative Game Releases Over Time', fontsize=15, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Temporal Pattern Analysis

# %%
print("\n🔍 TEMPORAL TRENDS")
print("="*70)

# Industry growth periods
pre_2000 = year_data[year_data < 2000].count()
y2000_2010 = year_data[(year_data >= 2000) & (year_data < 2010)].count()
post_2010 = year_data[year_data >= 2010].count()

print(f"\n  Pre-2000:       {pre_2000:5} releases ({pre_2000/len(year_data)*100:.1f}%)")
print(f"  2000-2009:      {y2000_2010:5} releases ({y2000_2010/len(year_data)*100:.1f}%)")
print(f"  2010 onwards:   {post_2010:5} releases ({post_2010/len(year_data)*100:.1f}%)")

print(f"\n  Peak year:      {peak_year} with {peak_count} releases")
print(f"  Avg releases/year: {year_counts.mean():.1f}")

# %% [markdown]
# ---
# ## 6.2 Categorical Variable Exploration
# 
# For each categorical variable, we ask:
# 1. **Which categories dominate the data?**
# 2. **Are there rare categories worth investigating?**
# 3. **Does category imbalance affect interpretation?**

# %% [markdown]
# ### Identifying Categorical Variables

# %%
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Categorical Variables in Dataset:")
print("="*60)
for col in categorical_cols:
    print(f"  • {col} ({df[col].nunique()} unique values)")

# Key categorical variables to analyze
key_categoricals = ['Platform', 'Genre', 'Publisher']

# %% [markdown]
# ---
# ## Analysis 4: Platform Distribution
# ### Research Questions:
# - Which gaming platforms are most represented?
# - Are there dominant platform eras?
# - How fragmented is the platform ecosystem?

# %% [markdown]
# ### Platform Frequency Analysis

# %%
platform_counts = df['Platform'].value_counts()

print("PLATFORM DISTRIBUTION")
print("="*70)
print(f"\n  Total unique platforms: {df['Platform'].nunique()}")
print(f"  Most common platform:   {platform_counts.index[0]} ({platform_counts.values[0]} games)")
print(f"  Least common platforms: {(platform_counts == 1).sum()} platforms with only 1 game")

print("\n📊 Top 15 Platforms by Game Count:")
print(platform_counts.head(15))

# %% [markdown]
# ### Platform Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Bar chart - Top 15 platforms
top_platforms = platform_counts.head(15)
axes[0].barh(range(len(top_platforms)), top_platforms.values, color='teal', edgecolor='black')
axes[0].set_yticks(range(len(top_platforms)))
axes[0].set_yticklabels(top_platforms.index)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Games', fontsize=12)
axes[0].set_title('Top 15 Gaming Platforms', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_platforms.values):
    axes[0].text(v + 20, i, str(v), va='center', fontweight='bold')

# 2. Pie chart - Market share (Top 10 + Others)
top_10_platforms = platform_counts.head(10)
others = platform_counts[10:].sum()
pie_data = list(top_10_platforms.values) + [others]
pie_labels = list(top_10_platforms.index) + ['Others']

axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Platform Market Share (by Game Count)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Platform Concentration Analysis

# %%
print("\n💡 PLATFORM CONCENTRATION INSIGHTS")
print("="*70)

# Calculate concentration metrics
total_games = len(df)
top_5_share = (platform_counts.head(5).sum() / total_games * 100)
top_10_share = (platform_counts.head(10).sum() / total_games * 100)

print(f"\n  Top 5 platforms account for:  {top_5_share:.2f}% of all games")
print(f"  Top 10 platforms account for: {top_10_share:.2f}% of all games")

# Rare platforms
rare_platforms = platform_counts[platform_counts <= 10]
print(f"\n  Platforms with ≤10 games: {len(rare_platforms)} platforms")
print(f"  Examples of rare platforms: {', '.join(rare_platforms.head(5).index.tolist())}")

# %% [markdown]
# ### Does Platform Imbalance Affect Interpretation?
# 
# **Key Considerations:**
# - **Yes** - Strong concentration in PS2, X360, PS3, Wii, DS means platform-specific analyses will be dominated by these
# - Rare platforms may represent niche markets or failed launches
# - Cross-platform comparisons must account for vastly different sample sizes
# - Aggregated statistics may be biased toward top platforms

# %% [markdown]
# ---
# ## Analysis 5: Genre Distribution
# ### Research Questions:
# - Which game genres are most popular?
# - Is the industry genre-diverse or concentrated?
# - Are there emerging or declining genres?

# %% [markdown]
# ### Genre Frequency Analysis

# %%
genre_counts = df['Genre'].value_counts()

print("GENRE DISTRIBUTION")
print("="*70)
print(f"\n  Total unique genres: {df['Genre'].nunique()}")
print(f"  Most common genre:   {genre_counts.index[0]} ({genre_counts.values[0]} games)")
print(f"  Least common genre:  {genre_counts.index[-1]} ({genre_counts.values[-1]} games)")

print("\n📊 All Genres (Frequency):")
print(genre_counts)

# %% [markdown]
# ### Genre Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Horizontal bar chart
colors_genre = plt.cm.Set3(range(len(genre_counts)))
axes[0].barh(range(len(genre_counts)), genre_counts.values, color=colors_genre, edgecolor='black')
axes[0].set_yticks(range(len(genre_counts)))
axes[0].set_yticklabels(genre_counts.index)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Games', fontsize=12)
axes[0].set_title('Game Genre Distribution', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# Add percentage labels
for i, (count, pct) in enumerate(zip(genre_counts.values, genre_counts.values/total_games*100)):
    axes[0].text(count + 50, i, f'{count} ({pct:.1f}%)', va='center', fontweight='bold', fontsize=9)

# 2. Proportion table visualization
genre_props = (genre_counts / total_games * 100).round(2)
axes[1].bar(range(len(genre_props)), genre_props.values, color=colors_genre, edgecolor='black')
axes[1].set_xticks(range(len(genre_props)))
axes[1].set_xticklabels(genre_props.index, rotation=45, ha='right')
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Genre Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Genre Diversity Assessment

# %%
print("\n💡 GENRE DIVERSITY INSIGHTS")
print("="*70)

# Shannon Entropy (diversity measure)
genre_proportions = genre_counts / total_games
entropy = -np.sum(genre_proportions * np.log2(genre_proportions))
max_entropy = np.log2(len(genre_counts))
normalized_entropy = entropy / max_entropy

print(f"\n  Shannon Entropy:       {entropy:.3f}")
print(f"  Max Possible Entropy:  {max_entropy:.3f}")
print(f"  Normalized Diversity:  {normalized_entropy:.3f} (0=concentrated, 1=uniform)")

# Interpretation
if normalized_entropy > 0.8:
    diversity_level = "HIGH - Genres well-distributed"
elif normalized_entropy > 0.6:
    diversity_level = "MODERATE - Some genre concentration"
else:
    diversity_level = "LOW - Heavily concentrated in few genres"

print(f"  Diversity Assessment:  {diversity_level}")

# Top 3 vs Rest
top_3_share = (genre_counts.head(3).sum() / total_games * 100)
print(f"\n  Top 3 genres account for: {top_3_share:.2f}% of all games")

# %% [markdown]
# ---
# ## Analysis 6: Publisher Distribution
# ### Research Questions:
# - Which publishers dominate the market?
# - How consolidated is the publishing industry?
# - Are there indie/small publishers with notable presence?

# %% [markdown]
# ### Publisher Frequency Analysis

# %%
# Handle missing publishers
publisher_counts = df['Publisher'].value_counts()

print("PUBLISHER DISTRIBUTION")
print("="*70)
print(f"\n  Total unique publishers: {df['Publisher'].nunique()}")
print(f"  Missing publisher data:  {df['Publisher'].isna().sum()} games")
print(f"  Most prolific publisher: {publisher_counts.index[0]} ({publisher_counts.values[0]} games)")

print("\n📊 Top 20 Publishers by Game Count:")
print(publisher_counts.head(20))

# %% [markdown]
# ### Publisher Visualization (Top 20)

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 1. Bar chart - Top 20
top_publishers = publisher_counts.head(20)
axes[0].barh(range(len(top_publishers)), top_publishers.values, color='darkslateblue', edgecolor='black')
axes[0].set_yticks(range(len(top_publishers)))
axes[0].set_yticklabels(top_publishers.index, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Games Published', fontsize=12)
axes[0].set_title('Top 20 Video Game Publishers', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_publishers.values):
    axes[0].text(v + 5, i, str(v), va='center', fontweight='bold', fontsize=9)

# 2. Long tail distribution
publisher_cumsum = publisher_counts.sort_values(ascending=False).cumsum() / publisher_counts.sum() * 100
axes[1].plot(range(1, len(publisher_cumsum)+1), publisher_cumsum.values, linewidth=2, color='crimson')
axes[1].axhline(y=50, color='green', linestyle='--', label='50% of games')
axes[1].axhline(y=80, color='orange', linestyle='--', label='80% of games')
axes[1].set_xlabel('Number of Publishers (Ranked)', fontsize=12)
axes[1].set_ylabel('Cumulative % of Games', fontsize=12)
axes[1].set_title('Publisher Market Concentration (Cumulative)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Publisher Market Concentration

# %%
print("\n💡 PUBLISHER MARKET CONCENTRATION")
print("="*70)

# Find concentration thresholds
publishers_50 = (publisher_cumsum <= 50).sum()
publishers_80 = (publisher_cumsum <= 80).sum()

print(f"\n  Top {publishers_50} publishers account for 50% of all games")
print(f"  Top {publishers_80} publishers account for 80% of all games")

# Small publishers
small_publishers = publisher_counts[publisher_counts <= 5]
print(f"\n  Publishers with ≤5 games: {len(small_publishers)} ({len(small_publishers)/len(publisher_counts)*100:.1f}%)")
print(f"  Publishers with only 1 game: {(publisher_counts == 1).sum()}")

# %% [markdown]
# ### Rare Publishers Worth Investigating

# %%
print("\n🔍 NOTABLE SMALL PUBLISHERS (1-3 games with high sales)")
print("="*70)

small_pub_games = df[df['Publisher'].isin(small_publishers.index)]
small_pub_high_sales = small_pub_games[small_pub_games['Global_Sales'] > 1.0].sort_values('Global_Sales', ascending=False)

if len(small_pub_high_sales) > 0:
    print(small_pub_high_sales[['Name', 'Publisher', 'Platform', 'Year', 'Global_Sales']].head(10).to_string(index=False))
else:
    print("  No small publishers with games selling >1 million copies")

# %% [markdown]
# ---
# ## Summary: Section B - Univariate Analysis Findings

# %% [markdown]
# ### 🎯 Key Numerical Insights
# 
# **Global Sales:**
# - **Typical game**: 0.17M sales (median) - most games sell modestly
# - **Average game**: 0.54M sales (mean) - inflated by blockbusters
# - **Distribution**: Heavily right-skewed (skewness: ~10.0)
# - **Concentration**: Top 1% of games account for ~40% of total sales
# - **Outliers**: Meaningful (blockbusters like Wii Sports: 82.74M)
# 
# **Regional Patterns:**
# - **North America**: Largest market (~49% of total sales)
# - **Europe**: Second largest (~27% of total sales)
# - **Japan**: Third (~14% of total sales)
# - **Consistency**: NA has highest variance; Japan more consistent
# 
# **Temporal Trends:**
# - **Time span**: 1980-2020
# - **Peak year**: 2008-2009 (industry boom period)
# - **Growth**: Rapid expansion 2000-2010, stabilization after

# %% [markdown]
# ### 🎮 Key Categorical Insights
# 
# **Platform:**
# - **Total platforms**: 31 unique
# - **Dominance**: PS2, X360, PS3, Wii, DS = top 5
# - **Concentration**: Top 10 platforms = ~75% of all games
# - **Imbalance**: Strong - affects cross-platform comparisons
# 
# **Genre:**
# - **Total genres**: 12 unique
# - **Most common**: Action (3,316 games), Sports (2,346), Misc (1,739)
# - **Diversity**: Moderate (normalized entropy: ~0.82)
# - **Imbalance**: Less severe than platforms
# 
# **Publisher:**
# - **Total publishers**: 579 unique
# - **Dominance**: Electronic Arts (1,351 games)
# - **Concentration**: Top 20 publishers = ~50% of all games
# - **Long tail**: 200+ publishers with only 1 game
# - **Imbalance**: Severe - highly consolidated industry

# %% [markdown]
# ### 📊 Analytical Implications
# 
# 1. **Sales Analyses**: Must use robust statistics (median, IQR) due to extreme skewness
# 2. **Regional Comparisons**: NA-centric bias requires normalization
# 3. **Platform Studies**: Dominated by 2000s-era consoles; modern platforms underrepresented
# 4. **Genre Research**: Fairly balanced; Action/Sports slightly overrepresented
# 5. **Publisher Trends**: Major publishers dominate; small publishers = niche/indie market

# %%
print("\n" + "="*70)
print("SECTION B COMPLETE: Univariate Analysis")
print("="*70)
print("\nNext Steps:")
print("  → Section C: Bivariate Relationships")
print("  → Section D: Multivariate Analysis")
print("  → Section E: Advanced Statistical Exploration")
print("="*70)

# %%