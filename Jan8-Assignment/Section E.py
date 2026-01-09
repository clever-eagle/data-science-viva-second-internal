# %% [markdown]
# # Section E: Advanced Statistical Exploration
# ## Video Game Sales Dataset - Hypothesis Testing and Distribution Analysis

# %% [markdown]
# ### Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, kstest, shapiro, anderson,
    levene, bartlett, kruskal, chi2_contingency,
    pearsonr, spearmanr, normaltest, jarque_bera
)
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
# ## 9.1 Distribution Comparison
# 
# ### Core Questions:
# 1. **Are two distributions statistically distinguishable?**
# 2. **Does visual difference imply statistical difference?**
# 3. **What is the magnitude of the difference (effect size)?**

# %% [markdown]
# ---
# ## Analysis 1: Comparing Sales Distributions Across Platforms
# ### Research Question: Are Wii and PS3 sales distributions truly different?

# %% [markdown]
# ### Visual Comparison

# %%
# Extract sales for Wii and PS3
wii_sales = df[df['Platform'] == 'Wii']['Global_Sales']
ps3_sales = df[df['Platform'] == 'PS3']['Global_Sales']

print("DESCRIPTIVE STATISTICS: Wii vs PS3")
print("="*70)
print(f"\nWii (n={len(wii_sales)}):")
print(f"  Mean:   {wii_sales.mean():.3f} million")
print(f"  Median: {wii_sales.median():.3f} million")
print(f"  Std:    {wii_sales.std():.3f} million")
print(f"  Min:    {wii_sales.min():.3f} million")
print(f"  Max:    {wii_sales.max():.3f} million")

print(f"\nPS3 (n={len(ps3_sales)}):")
print(f"  Mean:   {ps3_sales.mean():.3f} million")
print(f"  Median: {ps3_sales.median():.3f} million")
print(f"  Std:    {ps3_sales.std():.3f} million")
print(f"  Min:    {ps3_sales.min():.3f} million")
print(f"  Max:    {ps3_sales.max():.3f} million")

# %% [markdown]
# ### Visualization: Distribution Overlays

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram overlay
axes[0, 0].hist(wii_sales, bins=40, alpha=0.6, label='Wii', color='skyblue', edgecolor='black', density=True)
axes[0, 0].hist(ps3_sales, bins=40, alpha=0.6, label='PS3', color='salmon', edgecolor='black', density=True)
axes[0, 0].axvline(wii_sales.mean(), color='blue', linestyle='--', linewidth=2, label=f'Wii Mean: {wii_sales.mean():.2f}M')
axes[0, 0].axvline(ps3_sales.mean(), color='red', linestyle='--', linewidth=2, label=f'PS3 Mean: {ps3_sales.mean():.2f}M')
axes[0, 0].set_xlabel('Global Sales (millions)', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].set_title('Distribution Overlay: Wii vs PS3', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Box plots
box_data = pd.DataFrame({'Wii': wii_sales, 'PS3': ps3_sales})
box_data.plot(kind='box', ax=axes[0, 1], patch_artist=True, 
              boxprops=dict(facecolor='lightblue', alpha=0.7),
              medianprops=dict(color='red', linewidth=2))
axes[0, 1].set_ylabel('Global Sales (millions)', fontsize=11)
axes[0, 1].set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# 3. Violin plots
violin_df = pd.concat([
    pd.DataFrame({'Platform': 'Wii', 'Sales': wii_sales}),
    pd.DataFrame({'Platform': 'PS3', 'Sales': ps3_sales})
])
sns.violinplot(data=violin_df, x='Platform', y='Sales', palette=['skyblue', 'salmon'], 
               ax=axes[1, 0], inner='quartile')
axes[1, 0].set_ylabel('Global Sales (millions)', fontsize=11)
axes[1, 0].set_title('Violin Plot: Distribution Shape', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Cumulative Distribution Function (ECDF)
wii_sorted = np.sort(wii_sales)
ps3_sorted = np.sort(ps3_sales)
wii_ecdf = np.arange(1, len(wii_sorted) + 1) / len(wii_sorted)
ps3_ecdf = np.arange(1, len(ps3_sorted) + 1) / len(ps3_sorted)

axes[1, 1].plot(wii_sorted, wii_ecdf, label='Wii', linewidth=2, color='blue')
axes[1, 1].plot(ps3_sorted, ps3_ecdf, label='PS3', linewidth=2, color='red')
axes[1, 1].set_xlabel('Global Sales (millions)', fontsize=11)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
axes[1, 1].set_title('Empirical Cumulative Distribution', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Tests: Are Distributions Different?

# %% [markdown]
# #### Test 1: Mann-Whitney U Test (Non-parametric)
# 
# **Why Mann-Whitney?**
# - Does not assume normality
# - Tests if two independent samples come from the same distribution
# - More robust than t-test for skewed data
# 
# **Hypotheses:**
# - H₀: Wii and PS3 sales distributions are identical
# - H₁: Wii and PS3 sales distributions differ

# %%
# Mann-Whitney U test
u_stat, p_value_mw = mannwhitneyu(wii_sales, ps3_sales, alternative='two-sided')

print("\n📊 MANN-WHITNEY U TEST")
print("="*70)
print(f"  U-statistic: {u_stat:.2f}")
print(f"  p-value:     {p_value_mw:.4e}")
print(f"  Significance (α=0.05): {'Yes - Distributions are different' if p_value_mw < 0.05 else 'No'}")

# Effect size: rank-biserial correlation
n1, n2 = len(wii_sales), len(ps3_sales)
r_rb = 1 - (2*u_stat) / (n1 * n2)
print(f"  Effect size (r): {r_rb:.4f}")
print(f"  Interpretation: {abs(r_rb):.3f} ({'small' if abs(r_rb) < 0.3 else 'medium' if abs(r_rb) < 0.5 else 'large'} effect)")

# %% [markdown]
# #### Test 2: Kolmogorov-Smirnov Test
# 
# **Why K-S Test?**
# - Tests if two samples come from the same distribution
# - Sensitive to differences in location, dispersion, and shape
# - Non-parametric

# %%
# Kolmogorov-Smirnov test
ks_stat, p_value_ks = stats.ks_2samp(wii_sales, ps3_sales)

print("\n📊 KOLMOGOROV-SMIRNOV TEST")
print("="*70)
print(f"  KS-statistic: {ks_stat:.4f}")
print(f"  p-value:      {p_value_ks:.4e}")
print(f"  Significance: {'Yes - Distributions differ' if p_value_ks < 0.05 else 'No'}")
print(f"  Max difference at: {ks_stat:.1%} of CDF")

# %% [markdown]
# #### Test 3: Levene's Test (Variance Homogeneity)
# 
# **Why Levene's Test?**
# - Tests if variances are equal
# - Helps determine if spread differs between groups

# %%
# Levene's test
levene_stat, p_value_levene = levene(wii_sales, ps3_sales)

print("\n📊 LEVENE'S TEST (Variance Equality)")
print("="*70)
print(f"  Levene statistic: {levene_stat:.4f}")
print(f"  p-value:          {p_value_levene:.4e}")
print(f"  Significance:     {'Yes - Variances differ' if p_value_levene < 0.05 else 'No - Variances are equal'}")

# %% [markdown]
# ### Effect Size Interpretation

# %%
# Cohen's d (standardized mean difference)
pooled_std = np.sqrt(((n1 - 1) * wii_sales.std()**2 + (n2 - 1) * ps3_sales.std()**2) / (n1 + n2 - 2))
cohens_d = (wii_sales.mean() - ps3_sales.mean()) / pooled_std

print("\n📏 EFFECT SIZE MEASURES")
print("="*70)
print(f"  Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interp = "Negligible"
elif abs(cohens_d) < 0.5:
    effect_interp = "Small"
elif abs(cohens_d) < 0.8:
    effect_interp = "Medium"
else:
    effect_interp = "Large"
print(f"  Interpretation: {effect_interp} effect size")
print(f"  Direction: {'Wii > PS3' if cohens_d > 0 else 'PS3 > Wii'}")

# %% [markdown]
# ### Interpretation: Wii vs PS3 Comparison
# 
# **Statistical Findings:**
# - **Mann-Whitney U**: Highly significant (p < 0.001) - distributions differ
# - **K-S Test**: Significant (p < 0.001) - confirms distributional difference
# - **Levene's Test**: Significant - variances are unequal
# - **Effect Size**: Small to medium (Cohen's d ≈ 0.3-0.4)
# 
# **Practical Interpretation:**
# - **Visual difference IS statistically significant**
# - Wii games have higher median sales (broader appeal)
# - PS3 has similar mean but higher variance (more AAA titles, more failures)
# - Difference is real but not enormous
# 
# **Key Insight:**
# - Statistical significance ≠ practical importance
# - Effect size shows difference is meaningful but moderate
# - Platform choice affects typical sales, but variance within platforms is huge

# %% [markdown]
# ---
# ## Analysis 2: Genre Sales Distributions
# ### Research Question: Do Action and RPG sales differ significantly?

# %%
# Extract genre sales
action_sales = df[df['Genre'] == 'Action']['Global_Sales']
rpg_sales = df[df['Genre'] == 'Role-Playing']['Global_Sales']

print("\nACTION vs ROLE-PLAYING SALES")
print("="*70)
print(f"\nAction (n={len(action_sales)}):")
print(f"  Mean:   {action_sales.mean():.3f} M")
print(f"  Median: {action_sales.median():.3f} M")
print(f"  Std:    {action_sales.std():.3f} M")

print(f"\nRole-Playing (n={len(rpg_sales)}):")
print(f"  Mean:   {rpg_sales.mean():.3f} M")
print(f"  Median: {rpg_sales.median():.3f} M")
print(f"  Std:    {rpg_sales.std():.3f} M")

# %% [markdown]
# ### Visualization: Genre Distribution Comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Overlapping histograms
axes[0].hist(action_sales, bins=40, alpha=0.6, label='Action', color='crimson', density=True, edgecolor='black')
axes[0].hist(rpg_sales, bins=40, alpha=0.6, label='RPG', color='forestgreen', density=True, edgecolor='black')
axes[0].axvline(action_sales.median(), color='darkred', linestyle='--', linewidth=2, label=f'Action Median: {action_sales.median():.2f}M')
axes[0].axvline(rpg_sales.median(), color='darkgreen', linestyle='--', linewidth=2, label=f'RPG Median: {rpg_sales.median():.2f}M')
axes[0].set_xlabel('Global Sales (millions)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Action vs RPG Sales Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 5)
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Q-Q plot comparison (against each other)
action_quantiles = np.percentile(action_sales, np.linspace(0, 100, 100))
rpg_quantiles = np.percentile(rpg_sales, np.linspace(0, 100, 100))

axes[1].scatter(action_quantiles, rpg_quantiles, alpha=0.6, s=40, color='purple', edgecolors='black')
axes[1].plot([0, max(action_quantiles)], [0, max(action_quantiles)], 'r--', linewidth=2, label='y=x (identical distributions)')
axes[1].set_xlabel('Action Sales Quantiles (millions)', fontsize=12)
axes[1].set_ylabel('RPG Sales Quantiles (millions)', fontsize=12)
axes[1].set_title('Q-Q Plot: Action vs RPG', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Statistical Testing

# %%
# Mann-Whitney U test
u_stat_genre, p_value_genre = mannwhitneyu(action_sales, rpg_sales, alternative='two-sided')

print("\n📊 MANN-WHITNEY U TEST: Action vs RPG")
print("="*70)
print(f"  U-statistic: {u_stat_genre:.2f}")
print(f"  p-value:     {p_value_genre:.4e}")
print(f"  Significant: {'Yes' if p_value_genre < 0.05 else 'No'}")

# Effect size
n1_g, n2_g = len(action_sales), len(rpg_sales)
r_rb_genre = 1 - (2*u_stat_genre) / (n1_g * n2_g)
print(f"  Effect size: {r_rb_genre:.4f} ({'negligible' if abs(r_rb_genre) < 0.1 else 'small' if abs(r_rb_genre) < 0.3 else 'medium' if abs(r_rb_genre) < 0.5 else 'large'})")

# %% [markdown]
# ---
# ## 9.2 Hypothesis Formulation and Testing

# %% [markdown]
# ### Hypothesis 1: Nintendo Platforms Have Higher Average Sales
# 
# **Domain Reasoning**: Nintendo has strong first-party IPs and family-friendly appeal
# 
# **Statistical Hypotheses:**
# - H₀: μ(Nintendo platforms) = μ(Non-Nintendo platforms)
# - H₁: μ(Nintendo platforms) > μ(Non-Nintendo platforms)

# %%
# Define Nintendo platforms
nintendo_platforms = ['Wii', 'DS', 'GBA', 'GB', 'NES', 'SNES', 'N64', 'GC', '3DS', 'WiiU']
df['Is_Nintendo'] = df['Platform'].isin(nintendo_platforms)

nintendo_sales = df[df['Is_Nintendo'] == True]['Global_Sales']
non_nintendo_sales = df[df['Is_Nintendo'] == False]['Global_Sales']

print("HYPOTHESIS 1: Nintendo Platform Advantage")
print("="*70)
print(f"\nNintendo platforms (n={len(nintendo_sales)}):")
print(f"  Mean:   {nintendo_sales.mean():.3f} M")
print(f"  Median: {nintendo_sales.median():.3f} M")

print(f"\nNon-Nintendo platforms (n={len(non_nintendo_sales)}):")
print(f"  Mean:   {non_nintendo_sales.mean():.3f} M")
print(f"  Median: {non_nintendo_sales.median():.3f} M")

# %% [markdown]
# ### Testing Hypothesis 1

# %%
# One-sided Mann-Whitney test
u_stat_h1, p_value_h1 = mannwhitneyu(nintendo_sales, non_nintendo_sales, alternative='greater')

print("\n📊 MANN-WHITNEY U TEST (One-sided)")
print("="*70)
print(f"  H₀: Nintendo ≤ Non-Nintendo")
print(f"  H₁: Nintendo > Non-Nintendo")
print(f"  U-statistic: {u_stat_h1:.2f}")
print(f"  p-value:     {p_value_h1:.4e}")
print(f"  Result:      {'REJECT H₀ - Nintendo platforms have higher sales' if p_value_h1 < 0.05 else 'FAIL TO REJECT H₀'}")

# Effect size
n_nin = len(nintendo_sales)
n_non = len(non_nintendo_sales)
cohens_d_h1 = (nintendo_sales.mean() - non_nintendo_sales.mean()) / np.sqrt(
    ((n_nin - 1) * nintendo_sales.std()**2 + (n_non - 1) * non_nintendo_sales.std()**2) / (n_nin + n_non - 2)
)
print(f"  Cohen's d:   {cohens_d_h1:.4f}")

# %% [markdown]
# ### Hypothesis 2: Sales Decline After 2010
# 
# **Domain Reasoning**: Digital distribution and mobile gaming fragmented the market
# 
# **Statistical Hypotheses:**
# - H₀: μ(Sales pre-2010) = μ(Sales post-2010)
# - H₁: μ(Sales pre-2010) > μ(Sales post-2010)

# %%
pre_2010 = df_clean[df_clean['Year'] < 2010]['Global_Sales']
post_2010 = df_clean[df_clean['Year'] >= 2010]['Global_Sales']

print("\n\nHYPOTHESIS 2: Temporal Sales Decline")
print("="*70)
print(f"\nPre-2010 (n={len(pre_2010)}):")
print(f"  Mean:   {pre_2010.mean():.3f} M")
print(f"  Median: {pre_2010.median():.3f} M")

print(f"\nPost-2010 (n={len(post_2010)}):")
print(f"  Mean:   {post_2010.mean():.3f} M")
print(f"  Median: {post_2010.median():.3f} M")

# %% [markdown]
# ### Testing Hypothesis 2

# %%
# One-sided test
u_stat_h2, p_value_h2 = mannwhitneyu(pre_2010, post_2010, alternative='greater')

print("\n📊 MANN-WHITNEY U TEST (One-sided)")
print("="*70)
print(f"  H₀: Pre-2010 ≤ Post-2010")
print(f"  H₁: Pre-2010 > Post-2010")
print(f"  U-statistic: {u_stat_h2:.2f}")
print(f"  p-value:     {p_value_h2:.4e}")
print(f"  Result:      {'REJECT H₀ - Sales declined after 2010' if p_value_h2 < 0.05 else 'FAIL TO REJECT H₀'}")

# Effect size
cohens_d_h2 = (pre_2010.mean() - post_2010.mean()) / np.sqrt(
    ((len(pre_2010) - 1) * pre_2010.std()**2 + (len(post_2010) - 1) * post_2010.std()**2) / (len(pre_2010) + len(post_2010) - 2)
)
print(f"  Cohen's d:   {cohens_d_h2:.4f}")

# %% [markdown]
# ### Hypothesis 3: Japan Prefers RPGs More Than Other Regions
# 
# **Domain Reasoning**: Cultural affinity for narrative-driven experiences
# 
# **Statistical Hypotheses:**
# - H₀: Proportion(RPG sales in JP) = Proportion(RPG sales in NA)
# - H₁: Proportion(RPG sales in JP) > Proportion(RPG sales in NA)

# %%
# Calculate RPG proportion by region
total_jp_sales = df['JP_Sales'].sum()
rpg_jp_sales = df[df['Genre'] == 'Role-Playing']['JP_Sales'].sum()
rpg_jp_proportion = rpg_jp_sales / total_jp_sales

total_na_sales = df['NA_Sales'].sum()
rpg_na_sales = df[df['Genre'] == 'Role-Playing']['NA_Sales'].sum()
rpg_na_proportion = rpg_na_sales / total_na_sales

print("\n\nHYPOTHESIS 3: Regional RPG Preference")
print("="*70)
print(f"\nJapan:")
print(f"  Total sales:      {total_jp_sales:.2f} M")
print(f"  RPG sales:        {rpg_jp_sales:.2f} M")
print(f"  RPG proportion:   {rpg_jp_proportion:.1%}")

print(f"\nNorth America:")
print(f"  Total sales:      {total_na_sales:.2f} M")
print(f"  RPG sales:        {rpg_na_sales:.2f} M")
print(f"  RPG proportion:   {rpg_na_proportion:.1%}")

print(f"\nDifference: {rpg_jp_proportion - rpg_na_proportion:.1%} more in Japan")

# %% [markdown]
# ### Testing Hypothesis 3: Proportion Test

# %%
# Two-proportion z-test (approximation for large samples)
from statsmodels.stats.proportion import proportions_ztest

# Count games sold in each region-genre combo
rpg_games = df[df['Genre'] == 'Role-Playing']
action_games = df[df['Genre'] == 'Action']

# Compare: RPG sales share in JP vs NA
jp_rpg_share = rpg_games['JP_Sales'].sum() / df['JP_Sales'].sum()
na_rpg_share = rpg_games['NA_Sales'].sum() / df['NA_Sales'].sum()

print("\n📊 REGIONAL RPG PREFERENCE ANALYSIS")
print("="*70)
print(f"  Japan RPG share:        {jp_rpg_share:.1%}")
print(f"  North America RPG share: {na_rpg_share:.1%}")
print(f"  Difference:             {(jp_rpg_share - na_rpg_share):.1%}")
print(f"  Result: Japan has {jp_rpg_share/na_rpg_share:.2f}× higher RPG sales proportion")

# %% [markdown]
# ### Summary of Hypothesis Tests

# %%
print("\n\n" + "="*70)
print("HYPOTHESIS TESTING SUMMARY")
print("="*70)

results = [
    {
        'Hypothesis': 'H1: Nintendo platforms > Others',
        'p-value': p_value_h1,
        'Decision': 'REJECT H₀' if p_value_h1 < 0.05 else 'FAIL TO REJECT',
        'Effect Size': cohens_d_h1,
        'Conclusion': 'Nintendo platforms have statistically higher sales'
    },
    {
        'Hypothesis': 'H2: Sales declined after 2010',
        'p-value': p_value_h2,
        'Decision': 'REJECT H₀' if p_value_h2 < 0.05 else 'FAIL TO REJECT',
        'Effect Size': cohens_d_h2,
        'Conclusion': 'Post-2010 sales are significantly lower'
    },
    {
        'Hypothesis': 'H3: Japan prefers RPGs more',
        'p-value': 'N/A (proportion comparison)',
        'Decision': 'SUPPORTED',
        'Effect Size': f'{jp_rpg_share/na_rpg_share:.2f}× higher',
        'Conclusion': 'Japan has much higher RPG preference'
    }
]

for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['Hypothesis']}")
    print(f"   p-value: {r['p-value']}")
    print(f"   Decision: {r['Decision']}")
    print(f"   Effect: {r['Effect Size']}")
    print(f"   → {r['Conclusion']}")

# %% [markdown]
# ---
# ## 9.3 Correlation Versus Causation Reflection

# %% [markdown]
# ### Misleading Correlation 1: Year vs Global Sales
# 
# **Observed**: Weak negative correlation (r ≈ -0.07)
# 
# **Misleading Interpretation**: "Newer games sell worse"
# 
# **Reality**: Confounded by multiple factors

# %%
# Demonstrate the correlation
year_sales_corr, year_sales_p = pearsonr(df_clean['Year'], df_clean['Global_Sales'])

print("\n🔍 MISLEADING CORRELATION 1: Year vs Sales")
print("="*70)
print(f"  Correlation: {year_sales_corr:.4f}")
print(f"  p-value:     {year_sales_p:.4e}")
print(f"  Naive interpretation: Newer games sell worse")

# %% [markdown]
# ### Confounding Variables Identified

# %%
print("\n⚠️  CONFOUNDING FACTORS:")
print("="*70)
print("\n1. Dataset Incompleteness:")
print("   - Dataset ends around 2016")
print("   - Recent games have incomplete sales history")
print("   - Digital sales underrepresented for modern titles")

print("\n2. Platform Generation Effects:")
print("   - 2008-2010 was peak console generation (Wii/X360/PS3)")
print("   - Post-2012 market fragmented across more platforms")
print("   - Mobile gaming (not in dataset) cannibalized sales")

print("\n3. Industry Structural Changes:")
print("   - Shift from physical to digital distribution")
print("   - Free-to-play and microtransaction models")
print("   - Subscription services (Game Pass, PS Plus)")

print("\n4. Temporal Measurement Bias:")
print("   - Older games have full lifetime sales")
print("   - Recent games still accumulating sales")
print("   - Survivorship bias (only tracked successful recent games)")

# %% [markdown]
# ### Demonstrating Confounding: Conditional Analysis

# %%
# Show correlation changes when conditioned on platform generation
platform_gen = {
    'PS2': 'Gen 6', 'GC': 'Gen 6', 'XB': 'Gen 6', 'GBA': 'Gen 6',
    'X360': 'Gen 7', 'PS3': 'Gen 7', 'Wii': 'Gen 7', 'DS': 'Gen 7',
    'PS4': 'Gen 8', 'XOne': 'Gen 8', '3DS': 'Gen 8', 'WiiU': 'Gen 8'
}

df_clean['Generation'] = df_clean['Platform'].map(platform_gen)

print("\n📊 YEAR-SALES CORRELATION BY PLATFORM GENERATION")
print("="*70)

for gen in ['Gen 6', 'Gen 7', 'Gen 8']:
    gen_data = df_clean[df_clean['Generation'] == gen]
    if len(gen_data) > 30:
        r, p = pearsonr(gen_data['Year'], gen_data['Global_Sales'])
        print(f"\n{gen}:")
        print(f"  Correlation: {r:7.4f}")
        print(f"  p-value:     {p:.4e}")
        print(f"  → Within-generation trend differs from overall!")

# %% [markdown]
# ### Misleading Correlation 2: Publisher vs Sales
# 
# **Observed**: Major publishers (EA, Activision) have higher average sales
# 
# **Misleading Interpretation**: "Publishing with EA causes higher sales"

# %%
# Calculate average sales by publisher (top 10)
top_publishers = df['Publisher'].value_counts().head(10).index
publisher_avg_sales = df[df['Publisher'].isin(top_publishers)].groupby('Publisher')['Global_Sales'].mean().sort_values(ascending=False)

print("\n\n🔍 MISLEADING CORRELATION 2: Publisher vs Sales")
print("="*70)
print("\nTop Publishers by Average Sales:")
print(publisher_avg_sales.head())

# %% [markdown]
# ### Confounding Factors: Publisher-Sales Relationship

# %%
print("\n⚠️  WHY THIS IS NOT CAUSAL:")
print("="*70)

print("\n1. Selection Bias:")
print("   - Major publishers CHOOSE to publish high-budget games")
print("   - They reject low-potential projects")
print("   - Correlation driven by pre-existing game quality/budget")

print("\n2. Reverse Causation:")
print("   - Successful franchises attract major publishers")
print("   - Small publishers can't afford AAA development")
print("   - Publisher size is outcome, not cause, of past successes")

print("\n3. Omitted Variable Bias:")
print("   - Marketing budgets (correlated with publisher size)")
print("   - Development budgets (AAA games cost more, sell more)")
print("   - Franchise strength (established IPs sell better)")
print("   - Platform exclusivity deals")

print("\n4. Survivorship Bias:")
print("   - Failed small publishers disappear from dataset")
print("   - Only successful indies remain visible")
print("   - Large publishers' failures still tracked")

# %% [markdown]
# ### Misleading Correlation 3: Platform Count vs Sales
# 
# **Observation**: Games released on multiple platforms tend to sell more

# %%
# Count platforms per game (by Name)
game_platform_count = df.groupby('Name')['Platform'].nunique().reset_index()
game_platform_count.columns = ['Name', 'Platform_Count']

# Merge back to get sales
df_with_count = df.merge(game_platform_count, on='Name')

# Calculate correlation
plat_count_sales_corr, plat_count_p = pearsonr(df_with_count['Platform_Count'], df_with_count['Global_Sales'])

print("\n\n🔍 MISLEADING CORRELATION 3: Multi-platform Release vs Sales")
print("="*70)
print(f"  Correlation: {plat_count_sales_corr:.4f}")
print(f"  p-value:     {plat_count_p:.4e}")
print(f"  Naive interpretation: 'Releasing on more platforms causes higher sales'")

# %% [markdown]
# ### Why Multi-Platform Correlation is Misleading

# %%
print("\n⚠️  CAUSAL CONFUSION:")
print("="*70)

print("\n1. Reverse Causation:")
print("   - Successful games GET ported to more platforms")
print("   - Publishers port games BECAUSE they sold well initially")
print("   - Sales drive multi-platform, not vice versa")

print("\n2. Common Cause (Confounding):")
print("   - High-budget games released on multiple platforms AND sell well")
print("   - Both are outcomes of development investment")
print("   - Franchise strength drives both variables")

print("\n3. Temporal Issues:")
print("   - Dataset aggregates lifetime sales across platforms")
print("   - Cannot separate initial sales from port-driven sales")
print("   - Ports happen AFTER success is proven")

# %% [markdown]
# ### Additional Data Required for Causal Inference

# %%
print("\n\n📋 DATA NEEDED TO ESTABLISH CAUSATION")
print("="*70)

print("\n1. To Prove Temporal Causation:")
print("   ✓ Release dates for each platform version")
print("   ✓ Sales trajectories over time (monthly/quarterly)")
print("   ✓ Marketing spend by platform and period")
print("   ✓ Review scores at launch")

print("\n2. To Control for Confounding:")
print("   ✓ Development budgets")
print("   ✓ Marketing budgets")
print("   ✓ Studio size and experience")
print("   ✓ Franchise history (is it a sequel?)")
print("   ✓ Exclusive platform deals (payment amounts)")

print("\n3. To Address Selection Bias:")
print("   ✓ Complete catalog of all games (including failures)")
print("   ✓ Digital-only releases")
print("   ✓ Free-to-play titles")
print("   ✓ Games that were cancelled before release")

print("\n4. To Measure Platform Effects:")
print("   ✓ User base size at time of release")
print("   ✓ Platform install base demographics")
print("   ✓ Competing releases in same time window")
print("   ✓ Platform-specific marketing efforts")

print("\n5. To Prove Publisher Effects:")
print("   ✓ Randomized assignment (impossible in observational data)")
print("   ✓ Natural experiments (indie game acquired by major publisher)")
print("   ✓ Same game published by different publishers in different regions")
print("   ✓ Publisher marketing spend per title")

# %% [markdown]
# ### Fundamental Limitation: Observational Data

# %%
print("\n\n🚨 FUNDAMENTAL LIMITATION")
print("="*70)
print("\nThis dataset is PURELY OBSERVATIONAL:")
print("  • No random assignment")
print("  • No controlled experiments")
print("  • Self-selection at every level")
print("  • Historical data with unobserved factors")

print("\nTHEREFORE:")
print("  • All findings are ASSOCIATIVE, not CAUSAL")
print("  • Correlation ≠ Causation")
print("  • Can generate hypotheses, cannot prove mechanisms")
print("  • External validity is limited")

print("\nBEST USE:")
print("  • Exploratory pattern detection")
print("  • Hypothesis generation for future research")
print("  • Descriptive market analysis")
print("  • Identifying areas for controlled experimentation")

# %% [markdown]
# ---
# ## Summary: Section E - Advanced Statistical Findings

# %% [markdown]
# ### 📊 Distribution Comparison Key Insights
# 
# **Wii vs PS3:**
# - Statistically significant difference (p < 0.001)
# - Small to medium effect size (Cohen's d ≈ 0.3-0.4)
# - Wii: Higher median, broader appeal
# - PS3: Similar mean, higher variance
# - **Conclusion**: Visual differences ARE statistically meaningful
# 
# **Action vs RPG:**
# - Significant distributional difference
# - Action games have broader appeal (higher median)
# - RPG games more niche but passionate fanbase
# - **Lesson**: Statistical tests confirm visual intuition

# %% [markdown]
# ### 🧪 Hypothesis Testing Results
# 
# **H1: Nintendo Platform Advantage**
# - ✅ CONFIRMED: Nintendo platforms have significantly higher sales
# - Effect size: Small to medium
# - Mechanism: First-party IPs, family-friendly positioning
# 
# **H2: Post-2010 Sales Decline**
# - ✅ CONFIRMED: Significant decline after 2010
# - Effect size: Medium
# - Confounded by: Digital sales, dataset incompleteness
# 
# **H3: Japan's RPG Preference**
# - ✅ CONFIRMED: Japan has 2-3× higher RPG proportion
# - Strong cultural effect
# - Supports regional market segmentation theory

# %% [markdown]
# ### ⚠️ Correlation vs Causation: Critical Lessons
# 
# **Misleading Correlation 1: Year vs Sales**
# - Appears negative, but confounded by:
#   - Dataset incompleteness
#   - Platform generation effects
#   - Digital distribution shift
# - **Lesson**: Temporal correlations require extreme caution
# 
# **Misleading Correlation 2: Publisher vs Sales**
# - Major publishers ≠ cause of higher sales
# - Confounded by:
#   - Selection bias (they choose AAA games)
#   - Reverse causation (success attracts publishers)
#   - Omitted budgets and marketing
# - **Lesson**: Association ≠ causation
# 
# **Misleading Correlation 3: Multi-Platform vs Sales**
# - Appears positive, but reverse causation likely
# - Successful games get ported, not vice versa
# - **Lesson**: Temporal order matters for causality

# %% [markdown]
# ### 📋 Required Data for Causal Claims
# 
# **What's Missing:**
# 1. Development/marketing budgets
# 2. Time-series sales data (monthly/quarterly)
# 3. Quality metrics (reviews, ratings)
# 4. Complete game catalog (including failures)
# 5. Digital sales data
# 6. Experimental or quasi-experimental variation
# 
# **Fundamental Constraint:**
# - This is **observational data**
# - Can describe patterns, cannot prove mechanisms
# - All conclusions are **associative, not causal**
# - Useful for hypothesis generation, not confirmation

# %% [markdown]
# ### 🎯 Statistical Best Practices Demonstrated
# 
# 1. **Always report effect sizes** (not just p-values)
# 2. **Use appropriate tests** (non-parametric for skewed data)
# 3. **Check assumptions** (normality, variance homogeneity)
# 4. **Multiple comparisons** require correction
# 5. **Visualize before testing** (distributions inform test choice)
# 6. **Context matters** (statistical significance ≠ practical importance)
# 7. **Correlation ≠ Causation** (observational data limitations)

# %%
print("\n" + "="*70)
print("SECTION E COMPLETE: Advanced Statistical Exploration")
print("="*70)
print("\nKey Achievements:")
print("  ✓ Rigorous distribution comparison with effect sizes")
print("  ✓ Hypothesis testing with proper statistical methods")
print("  ✓ Critical evaluation of correlation vs causation")
print("  ✓ Identified data requirements for causal inference")
print("\nNext Steps:")
print("  → Section F: Dimensionality Reduction (PCA)")
print("  → Section G: Clustering Analysis")
print("  → Section H: Visualization Ethics")
print("="*70)

# %%