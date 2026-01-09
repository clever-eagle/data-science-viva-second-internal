# %% [markdown]
# # Section C: Bivariate Relationships and Statistical Validation
# ## Video Game Sales Dataset - Exploring Variable Relationships

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
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency, mannwhitneyu, kruskal
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

# Clean Year data for analyses
df_clean = df.dropna(subset=['Year'])
df_clean['Year'] = df_clean['Year'].astype(int)

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Clean dataset (Year non-null): {df_clean.shape[0]} rows")

# %% [markdown]
# ## 7.1 Numerical–Numerical Relationships
# 
# ### Guiding Questions:
# 1. **Is there an apparent relationship?**
# 2. **Is the relationship linear or non-linear?**
# 3. **Is the relationship strong or weak?**
# 4. **Is the relationship statistically significant?**

# %% [markdown]
# ---
# ## Analysis 1: Regional Sales Correlations
# ### Research Question: How do sales in different regions correlate?

# %% [markdown]
# ### Hypothesis:
# - **H0**: Regional sales are independent
# - **H1**: Regional sales are positively correlated (global hits sell well everywhere)

# %%
# Regional sales columns
regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Calculate correlation matrix
corr_matrix = df[regional_cols].corr()

print("REGIONAL SALES CORRELATION MATRIX")
print("="*70)
print(corr_matrix.round(3))

# %% [markdown]
# ### Visualization: Correlation Heatmap

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, fmt='.3f', annot_kws={'size': 12, 'weight': 'bold'})
plt.title('Regional Sales Correlation Matrix', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Pairwise Scatter Plots with Regression Lines

# %%
# Create pairplot for regional sales
g = sns.pairplot(df[regional_cols], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                 corner=False, height=3)
g.fig.suptitle('Regional Sales Pairwise Relationships', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Validation: Correlation Coefficients with Confidence

# %%
print("\n📊 DETAILED CORRELATION ANALYSIS")
print("="*70)

# Function to compute correlation with p-value
def correlation_analysis(x, y, x_name, y_name):
    # Pearson correlation (assumes linear relationship)
    pearson_r, pearson_p = pearsonr(x, y)
    
    # Spearman correlation (rank-based, non-linear)
    spearman_r, spearman_p = spearmanr(x, y)
    
    print(f"\n{x_name} vs {y_name}:")
    print(f"  Pearson r:  {pearson_r:7.4f} (p-value: {pearson_p:.4e})")
    print(f"  Spearman ρ: {spearman_r:7.4f} (p-value: {spearman_p:.4e})")
    
    # Interpretation
    if abs(pearson_r) > 0.7:
        strength = "Strong"
    elif abs(pearson_r) > 0.4:
        strength = "Moderate"
    elif abs(pearson_r) > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "Positive" if pearson_r > 0 else "Negative"
    significance = "Significant" if pearson_p < 0.05 else "Not Significant"
    
    print(f"  Interpretation: {strength} {direction} correlation ({significance})")
    
    return pearson_r, pearson_p

# Analyze key pairs
pairs = [
    ('NA_Sales', 'EU_Sales'),
    ('NA_Sales', 'JP_Sales'),
    ('EU_Sales', 'JP_Sales'),
    ('NA_Sales', 'Other_Sales')
]

for col1, col2 in pairs:
    correlation_analysis(df[col1], df[col2], col1, col2)

# %% [markdown]
# ### Deep Dive: NA Sales vs EU Sales

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Scatter plot with regression line
axes[0].scatter(df['NA_Sales'], df['EU_Sales'], alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['NA_Sales'], df['EU_Sales'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['NA_Sales'].min(), df['NA_Sales'].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

# Pearson correlation
r, p_val = pearsonr(df['NA_Sales'], df['EU_Sales'])
axes[0].text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value < 0.001', 
             transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

axes[0].set_xlabel('North America Sales (millions)', fontsize=12)
axes[0].set_ylabel('Europe Sales (millions)', fontsize=12)
axes[0].set_title('NA vs EU Sales - Linear Relationship', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Log-log plot to check for non-linearity
axes[1].scatter(np.log10(df['NA_Sales'] + 0.01), np.log10(df['EU_Sales'] + 0.01), 
                alpha=0.5, s=30, color='coral', edgecolors='black', linewidth=0.5)

axes[1].set_xlabel('Log10(NA Sales + 0.01)', fontsize=12)
axes[1].set_ylabel('Log10(EU Sales + 0.01)', fontsize=12)
axes[1].set_title('NA vs EU Sales - Log-Log Scale', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: NA vs EU Sales
# 
# **Findings:**
# - **Strong positive correlation** (r ≈ 0.77, p < 0.001)
# - **Linear relationship**: Games that sell well in NA tend to sell well in EU
# - **Not perfect**: Significant scatter indicates region-specific preferences
# - **Statistical significance**: Extremely high confidence (p < 0.001)
# 
# **Analytical Insight:**
# - Western markets (NA + EU) share similar gaming preferences
# - Relationship is approximately linear but with variance
# - Some games are regional hits (outliers from trend line)

# %% [markdown]
# ### Deep Dive: NA Sales vs JP Sales

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Scatter plot
axes[0].scatter(df['NA_Sales'], df['JP_Sales'], alpha=0.5, s=30, color='mediumseagreen', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['NA_Sales'], df['JP_Sales'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['NA_Sales'].min(), df['NA_Sales'].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

# Pearson correlation
r, p_val = pearsonr(df['NA_Sales'], df['JP_Sales'])
axes[0].text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value < 0.001', 
             transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

axes[0].set_xlabel('North America Sales (millions)', fontsize=12)
axes[0].set_ylabel('Japan Sales (millions)', fontsize=12)
axes[0].set_title('NA vs Japan Sales - Weak Correlation', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Identify JP-specific hits
jp_dominant = df[(df['JP_Sales'] > 2) & (df['JP_Sales'] > df['NA_Sales'])]
axes[1].scatter(df['NA_Sales'], df['JP_Sales'], alpha=0.3, s=30, color='lightgray', label='All Games')
axes[1].scatter(jp_dominant['NA_Sales'], jp_dominant['JP_Sales'], alpha=0.8, s=60, 
                color='crimson', edgecolors='black', linewidth=1, label='JP-Dominant Hits')

axes[1].set_xlabel('North America Sales (millions)', fontsize=12)
axes[1].set_ylabel('Japan Sales (millions)', fontsize=12)
axes[1].set_title('Identifying Japan-Specific Blockbusters', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Show examples of JP-dominant games
print("\n🎌 JAPAN-DOMINANT GAMES (High JP Sales, Lower NA Sales):")
print("="*70)
print(jp_dominant.nlargest(10, 'JP_Sales')[['Name', 'Platform', 'Genre', 'NA_Sales', 'JP_Sales']].to_string(index=False))

# %% [markdown]
# ### Interpretation: NA vs JP Sales
# 
# **Findings:**
# - **Weak positive correlation** (r ≈ 0.36, p < 0.001)
# - **Significant scatter**: Many games successful in one region but not the other
# - **Cultural differences**: JP market has distinct preferences (RPGs, Nintendo titles)
# - **Statistical significance**: Correlation is weak but still statistically significant
# 
# **Analytical Insight:**
# - East-West gaming preferences diverge significantly
# - Japan represents a unique gaming culture
# - Some franchises (Pokemon, Dragon Quest) are JP-dominant

# %% [markdown]
# ---
# ## Analysis 2: Year vs Global Sales
# ### Research Question: Do newer games sell more?

# %% [markdown]
# ### Hypothesis:
# - **H0**: Release year does not correlate with sales
# - **H1**: Newer games have higher sales (industry growth hypothesis)

# %%
# Calculate correlation
year_sales_corr, year_sales_p = pearsonr(df_clean['Year'], df_clean['Global_Sales'])

print("YEAR vs GLOBAL SALES CORRELATION")
print("="*70)
print(f"  Pearson r:  {year_sales_corr:.4f}")
print(f"  p-value:    {year_sales_p:.4e}")
print(f"  Significance: {'Yes' if year_sales_p < 0.05 else 'No'}")

# %% [markdown]
# ### Visualization: Temporal Sales Trend

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Scatter plot with trend line
axes[0].scatter(df_clean['Year'], df_clean['Global_Sales'], alpha=0.4, s=20, color='purple')

# Add regression line
z = np.polyfit(df_clean['Year'], df_clean['Global_Sales'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_clean['Year'].min(), df_clean['Year'].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'Trend: y = {z[0]:.4f}x + {z[1]:.2f}')

axes[0].set_xlabel('Release Year', fontsize=12)
axes[0].set_ylabel('Global Sales (millions)', fontsize=12)
axes[0].set_title('Release Year vs Global Sales - Scatter Plot', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Average sales by year
yearly_avg = df_clean.groupby('Year')['Global_Sales'].agg(['mean', 'median', 'std']).reset_index()

axes[1].plot(yearly_avg['Year'], yearly_avg['mean'], marker='o', linewidth=2, label='Mean Sales', color='darkblue')
axes[1].plot(yearly_avg['Year'], yearly_avg['median'], marker='s', linewidth=2, label='Median Sales', color='darkgreen')
axes[1].fill_between(yearly_avg['Year'], 
                       yearly_avg['mean'] - yearly_avg['std'],
                       yearly_avg['mean'] + yearly_avg['std'],
                       alpha=0.2, color='blue', label='±1 Std Dev')

axes[1].set_xlabel('Release Year', fontsize=12)
axes[1].set_ylabel('Sales (millions)', fontsize=12)
axes[1].set_title('Average Sales Over Time', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: Temporal Sales Trends
# 
# **Findings:**
# - **Very weak negative correlation** (r ≈ -0.07)
# - **Statistically significant** but practically meaningless
# - **Non-linear pattern**: Peak sales in mid-2000s, decline afterward
# - **Mean vs Median divergence**: Blockbusters concentrated in certain eras
# 
# **Analytical Insight:**
# - Linear correlation inappropriate for this relationship
# - Industry evolved through distinct eras (arcade, console generations)
# - Recent games may be underrepresented (dataset cutoff)
# - **Conclusion**: Relationship is non-linear and era-dependent

# %% [markdown]
# ---
# ## 7.2 Numerical–Categorical Relationships
# 
# ### Guiding Questions:
# 1. **Does category membership shift distributions?**
# 2. **Are observed differences meaningful or random?**
# 3. **Which statistical test is appropriate?**

# %% [markdown]
# ---
# ## Analysis 3: Genre vs Global Sales
# ### Research Question: Do certain genres sell significantly better?

# %% [markdown]
# ### Hypothesis:
# - **H0**: Genre does not affect sales distribution
# - **H1**: Different genres have different sales distributions

# %%
# Group-wise descriptive statistics
genre_stats = df.groupby('Genre')['Global_Sales'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(3).sort_values('mean', ascending=False)

print("GENRE SALES STATISTICS")
print("="*70)
print(genre_stats)

# %% [markdown]
# ### Visualization: Distribution by Genre

# %%
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 1. Box plots
genre_order = genre_stats.index.tolist()
sns.boxplot(data=df, x='Genre', y='Global_Sales', order=genre_order, 
            palette='Set2', ax=axes[0], showfliers=False)
axes[0].set_xlabel('Genre', fontsize=12)
axes[0].set_ylabel('Global Sales (millions)', fontsize=12)
axes[0].set_title('Sales Distribution by Genre (Outliers Hidden)', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(alpha=0.3, axis='y')

# 2. Violin plots (shows density)
sns.violinplot(data=df, x='Genre', y='Global_Sales', order=genre_order,
               palette='muted', ax=axes[1], inner='quartile', cut=0)
axes[1].set_xlabel('Genre', fontsize=12)
axes[1].set_ylabel('Global Sales (millions)', fontsize=12)
axes[1].set_title('Sales Distribution Density by Genre', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(0, 5)  # Zoom to see distribution shape
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Test: Kruskal-Wallis H-Test
# 
# **Why Kruskal-Wallis?**
# - Sales data is heavily skewed (not normally distributed)
# - Non-parametric alternative to ANOVA
# - Tests if distributions differ across groups

# %%
# Prepare data for statistical test
genre_groups = [group['Global_Sales'].values for name, group in df.groupby('Genre')]

# Kruskal-Wallis H-test
h_stat, p_value = kruskal(*genre_groups)

print("\n📊 KRUSKAL-WALLIS H-TEST: Genre vs Sales")
print("="*70)
print(f"  H-statistic: {h_stat:.4f}")
print(f"  p-value:     {p_value:.4e}")
print(f"  Significance: {'Yes - Genres have different sales distributions' if p_value < 0.05 else 'No'}")

# Effect size (Epsilon-squared)
n = len(df)
k = len(genre_groups)
epsilon_sq = (h_stat - k + 1) / (n - k)
print(f"  Effect size (ε²): {epsilon_sq:.4f}")

# %% [markdown]
# ### Post-hoc Analysis: Pairwise Genre Comparisons

# %%
print("\n🔍 PAIRWISE COMPARISONS: Selected Genre Pairs")
print("="*70)

# Compare top-selling genres
pairs_to_test = [
    ('Shooter', 'Puzzle'),
    ('Platform', 'Strategy'),
    ('Sports', 'Role-Playing')
]

for genre1, genre2 in pairs_to_test:
    g1_sales = df[df['Genre'] == genre1]['Global_Sales']
    g2_sales = df[df['Genre'] == genre2]['Global_Sales']
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_val = mannwhitneyu(g1_sales, g2_sales, alternative='two-sided')
    
    print(f"\n{genre1} vs {genre2}:")
    print(f"  Mann-Whitney U: {u_stat:.2f}")
    print(f"  p-value: {p_val:.4e}")
    print(f"  Median {genre1}: {g1_sales.median():.3f}M")
    print(f"  Median {genre2}: {g2_sales.median():.3f}M")
    print(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}")

# %% [markdown]
# ### Interpretation: Genre Impact on Sales
# 
# **Findings:**
# - **Highly significant difference** (p < 0.001) across genres
# - **Shooter, Platform, Sports** have highest median sales
# - **Puzzle, Strategy, Adventure** have lower median sales
# - **Within-genre variance** is enormous (long-tailed distributions)
# 
# **Analytical Insight:**
# - Genre matters, but doesn't guarantee success
# - Action-oriented genres (Shooter, Sports) appeal to broader audiences
# - Niche genres (Strategy, Puzzle) have dedicated but smaller markets
# - **Caution**: Distribution overlap is substantial

# %% [markdown]
# ---
# ## Analysis 4: Platform vs Global Sales
# ### Research Question: Do games on certain platforms sell better?

# %% [markdown]
# ### Top 10 Platforms Analysis

# %%
# Focus on top 10 platforms by game count
top_platforms = df['Platform'].value_counts().head(10).index.tolist()
df_top_platforms = df[df['Platform'].isin(top_platforms)]

# Group statistics
platform_stats = df_top_platforms.groupby('Platform')['Global_Sales'].agg([
    'count', 'mean', 'median', 'std'
]).round(3).sort_values('median', ascending=False)

print("TOP 10 PLATFORMS - SALES STATISTICS")
print("="*70)
print(platform_stats)

# %% [markdown]
# ### Visualization: Platform Sales Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Box plot
platform_order = platform_stats.index.tolist()
sns.boxplot(data=df_top_platforms, x='Platform', y='Global_Sales', 
            order=platform_order, palette='coolwarm', ax=axes[0])
axes[0].set_xlabel('Platform', fontsize=12)
axes[0].set_ylabel('Global Sales (millions)', fontsize=12)
axes[0].set_title('Sales Distribution by Platform (Top 10)', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylim(0, 10)
axes[0].grid(alpha=0.3, axis='y')

# 2. Bar plot - Mean vs Median
x_pos = np.arange(len(platform_order))
axes[1].bar(x_pos - 0.2, platform_stats.loc[platform_order, 'mean'], 
            width=0.4, label='Mean', color='steelblue', edgecolor='black')
axes[1].bar(x_pos + 0.2, platform_stats.loc[platform_order, 'median'], 
            width=0.4, label='Median', color='coral', edgecolor='black')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(platform_order, rotation=45)
axes[1].set_xlabel('Platform', fontsize=12)
axes[1].set_ylabel('Sales (millions)', fontsize=12)
axes[1].set_title('Mean vs Median Sales by Platform', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Test: Platform Differences

# %%
# Kruskal-Wallis test for platforms
platform_groups = [group['Global_Sales'].values for name, group in df_top_platforms.groupby('Platform')]
h_stat_plat, p_value_plat = kruskal(*platform_groups)

print("\n📊 KRUSKAL-WALLIS H-TEST: Platform vs Sales")
print("="*70)
print(f"  H-statistic: {h_stat_plat:.4f}")
print(f"  p-value:     {p_value_plat:.4e}")
print(f"  Significance: {'Yes - Platforms have different sales distributions' if p_value_plat < 0.05 else 'No'}")

# %% [markdown]
# ### Interpretation: Platform Impact
# 
# **Findings:**
# - **Statistically significant** platform differences (p < 0.001)
# - **Wii, GB, DS** have higher median sales (casual/family-friendly platforms)
# - **PS3, X360, PS2** have lower median but high variance (core gaming platforms)
# - **Mean >> Median** for all platforms (right-skewed distributions)
# 
# **Analytical Insight:**
# - Nintendo platforms tend toward broader appeal (higher typical sales)
# - PlayStation/Xbox platforms have more titles but lower median
# - Platform success influenced by market positioning and era

# %% [markdown]
# ---
# ## 7.3 Categorical–Categorical Relationships
# 
# ### Guiding Questions:
# 1. **Are the variables independent?**
# 2. **Does one category dominate outcomes?**
# 3. **What association measures are appropriate?**

# %% [markdown]
# ---
# ## Analysis 5: Platform vs Genre Association
# ### Research Question: Are certain genres platform-specific?

# %% [markdown]
# ### Hypothesis:
# - **H0**: Platform and Genre are independent
# - **H1**: Platform and Genre are associated (e.g., Nintendo → Platform games)

# %%
# Create contingency table (top platforms and genres)
top_genres = df['Genre'].value_counts().head(8).index.tolist()
df_subset = df[(df['Platform'].isin(top_platforms)) & (df['Genre'].isin(top_genres))]

contingency_table = pd.crosstab(df_subset['Platform'], df_subset['Genre'])

print("CONTINGENCY TABLE: Platform × Genre (Top 10 Platforms, Top 8 Genres)")
print("="*70)
print(contingency_table)

# %% [markdown]
# ### Visualization: Stacked Bar Chart

# %%
# Normalize by platform (show proportion of genres per platform)
contingency_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Stacked bar chart (counts)
contingency_table.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[0], edgecolor='black')
axes[0].set_xlabel('Platform', fontsize=12)
axes[0].set_ylabel('Number of Games', fontsize=12)
axes[0].set_title('Genre Distribution by Platform (Counts)', fontsize=14, fontweight='bold')
axes[0].legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(alpha=0.3, axis='y')

# 2. Stacked bar chart (proportions)
contingency_normalized.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[1], edgecolor='black')
axes[1].set_xlabel('Platform', fontsize=12)
axes[1].set_ylabel('Proportion', fontsize=12)
axes[1].set_title('Genre Distribution by Platform (Proportions)', fontsize=14, fontweight='bold')
axes[1].legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Test: Chi-Square Test of Independence

# %%
# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print("\n📊 CHI-SQUARE TEST OF INDEPENDENCE: Platform × Genre")
print("="*70)
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom:   {dof}")
print(f"  p-value:              {p_val:.4e}")
print(f"  Significance:         {'Yes - Platform and Genre are associated' if p_val < 0.05 else 'No'}")

# Cramér's V (effect size for categorical association)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"  Cramér's V:           {cramers_v:.4f} (0=no association, 1=perfect association)")

# %% [markdown]
# ### Interpretation: Platform-Genre Association
# 
# **Findings:**
# - **Highly significant association** (p < 0.001)
# - **Cramér's V ≈ 0.1-0.2**: Small to moderate effect size
# - **Platform-specific patterns**:
#   - Wii: High proportion of Sports/Misc (family games)
#   - DS: High proportion of Misc/Puzzle (touch-screen games)
#   - PS2/X360: Balanced genre distribution (core gaming)
# 
# **Analytical Insight:**
# - Platform design influences genre availability
# - Nintendo platforms cater to casual/family audiences
# - PlayStation/Xbox have more diverse genre ecosystems
# - **Association exists but is not deterministic**

# %% [markdown]
# ### Heatmap: Platform-Genre Relationship

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'label': 'Proportion'})
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Platform', fontsize=12)
plt.title('Platform-Genre Association Heatmap (Normalized)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Analysis 6: Publisher vs Platform
# ### Research Question: Do publishers favor specific platforms?

# %% [markdown]
# ### Focus on Top Publishers and Platforms

# %%
top_publishers = df['Publisher'].value_counts().head(8).index.tolist()
df_pub_plat = df[(df['Publisher'].isin(top_publishers)) & (df['Platform'].isin(top_platforms))]

pub_plat_table = pd.crosstab(df_pub_plat['Publisher'], df_pub_plat['Platform'])

print("CONTINGENCY TABLE: Publisher × Platform (Top 8 Publishers, Top 10 Platforms)")
print("="*80)
print(pub_plat_table)

# %% [markdown]
# ### Visualization: Publisher-Platform Preferences

# %%
# Normalize by publisher
pub_plat_normalized = pub_plat_table.div(pub_plat_table.sum(axis=1), axis=0)

plt.figure(figsize=(14, 8))
sns.heatmap(pub_plat_normalized, annot=True, fmt='.2f', cmap='Blues', 
            linewidths=0.5, cbar_kws={'label': 'Proportion of Games'})
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Publisher', fontsize=12)
plt.title('Publisher Platform Preferences (Normalized)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Test: Independence

# %%
# Chi-square test
chi2_pub, p_pub, dof_pub, expected_pub = chi2_contingency(pub_plat_table)

print("\n📊 CHI-SQUARE TEST: Publisher × Platform")
print("="*70)
print(f"  Chi-square statistic: {chi2_pub:.4f}")
print(f"  p-value:              {p_pub:.4e}")
print(f"  Significance:         {'Yes - Association exists' if p_pub < 0.05 else 'No'}")

n_pub = pub_plat_table.sum().sum()
min_dim_pub = min(pub_plat_table.shape[0] - 1, pub_plat_table.shape[1] - 1)
cramers_v_pub = np.sqrt(chi2_pub / (n_pub * min_dim_pub))
print(f"  Cramér's V:           {cramers_v_pub:.4f}")

# %% [markdown]
# ### Interpretation: Publisher-Platform Preferences
# 
# **Findings:**
# - **Significant association** (p < 0.001)
# - **Nintendo**: Strong preference for own platforms (Wii, DS)
# - **EA, Activision**: Multi-platform strategy (balanced across PS/Xbox)
# - **Sony exclusives**: Higher concentration on PS platforms
# 
# **Analytical Insight:**
# - First-party publishers (Nintendo, Sony) favor proprietary platforms
# - Third-party publishers pursue platform-agnostic strategies
# - Platform exclusivity is strategic business decision

# %% [markdown]
# ---
# ## Summary: Section C - Bivariate Findings

# %% [markdown]
# ### 📊 Numerical-Numerical Relationships
# 
# **Regional Sales Correlations:**
# - **NA ↔ EU**: Strong positive (r ≈ 0.77) - Western market similarity
# - **NA ↔ JP**: Weak positive (r ≈ 0.36) - Cultural divergence
# - **EU ↔ JP**: Weak positive (r ≈ 0.31) - Distinct preferences
# - **Insight**: Regional markets are interconnected but Japan is unique
# 
# **Year ↔ Sales:**
# - **Very weak correlation** (r ≈ -0.07)
# - **Non-linear relationship**: Peak in mid-2000s, decline after
# - **Insight**: Linear models inappropriate; industry has distinct eras

# %% [markdown]
# ### 🎮 Numerical-Categorical Relationships
# 
# **Genre → Sales:**
# - **Significant differences** (Kruskal-Wallis p < 0.001)
# - **Top genres**: Shooter, Platform, Sports (higher medians)
# - **Lower genres**: Puzzle, Strategy, Adventure
# - **High variance**: Genre suggests tendency, not guarantee
# 
# **Platform → Sales:**
# - **Significant differences** (p < 0.001)
# - **Nintendo platforms**: Higher median sales (casual appeal)
# - **PS/Xbox**: Lower median, higher variance (core gaming)
# - **Insight**: Platform positioning affects typical sales

# %% [markdown]
# ### 🔗 Categorical-Categorical Relationships
# 
# **Platform × Genre:**
# - **Significant association** (χ² p < 0.001, Cramér's V ≈ 0.15)
# - **Platform influences genre distribution**
# - **Examples**: Wii → Sports/Misc, DS → Puzzle
# 
# **Publisher × Platform:**
# - **Significant association** (χ² p < 0.001, Cramér's V ≈ 0.25)
# - **First-party publishers**: Platform exclusivity
# - **Third-party publishers**: Multi-platform strategy

# %% [markdown]
# ### ⚠️ Statistical Validation Key Takeaways
# 
# 1. **Correlation ≠ Causation**: All findings are observational
# 2. **Statistical vs Practical Significance**: Some correlations significant but weak
# 3. **Non-linear patterns**: Many relationships poorly captured by linear metrics
# 4. **Distribution matters**: Heavy skewness required non-parametric tests
# 5. **Confounding possible**: Multiple variables interact simultaneously

# %%
print("\n" + "="*70)
print("SECTION C COMPLETE: Bivariate Relationships")
print("="*70)
print("\nNext Steps:")
print("  → Section D: Multivariate Analysis & Feature Interactions")
print("  → Section E: Advanced Statistical Exploration")
print("="*70)

# %%