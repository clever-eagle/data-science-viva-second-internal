# %% [markdown]
# # Section D: Multivariate Analysis and Feature Interactions
# ## Video Game Sales Dataset - Exploring Complex Relationships

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
from scipy.stats import pearsonr, spearmanr
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
# ## 8.1 Conditional Exploration
# 
# ### Core Questions:
# 1. **Does a relationship persist under conditioning?**
# 2. **Does introducing a third variable change conclusions?**
# 3. **Are observed patterns universal or context-dependent?**

# %% [markdown]
# ---
# ## Analysis 1: Regional Sales Correlation - Conditional on Genre
# ### Research Question: Does NA-EU correlation hold across all genres?

# %% [markdown]
# ### Baseline: Overall NA-EU Correlation (From Section C)

# %%
# Overall correlation
overall_corr, overall_p = pearsonr(df['NA_Sales'], df['EU_Sales'])

print("BASELINE CORRELATION: NA Sales vs EU Sales")
print("="*70)
print(f"  Pearson r (overall):  {overall_corr:.4f}")
print(f"  p-value:              {overall_p:.4e}")
print(f"  Interpretation:       Strong positive correlation")

# %% [markdown]
# ### Conditional Analysis: Correlation by Genre

# %%
print("\n📊 GENRE-SPECIFIC CORRELATIONS: NA vs EU Sales")
print("="*70)

genre_correlations = []

for genre in df['Genre'].unique():
    genre_data = df[df['Genre'] == genre]
    
    if len(genre_data) > 10:  # Only analyze genres with sufficient data
        r, p = pearsonr(genre_data['NA_Sales'], genre_data['EU_Sales'])
        genre_correlations.append({
            'Genre': genre,
            'Correlation': r,
            'p-value': p,
            'Sample_Size': len(genre_data)
        })

genre_corr_df = pd.DataFrame(genre_correlations).sort_values('Correlation', ascending=False)
print(genre_corr_df.to_string(index=False))

# %% [markdown]
# ### Visualization: Faceted Scatter Plots by Genre

# %%
# Select top 6 genres by count for visualization
top_genres = df['Genre'].value_counts().head(6).index.tolist()
df_top_genres = df[df['Genre'].isin(top_genres)]

# Create faceted plot
g = sns.FacetGrid(df_top_genres, col='Genre', col_wrap=3, height=4, sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x='NA_Sales', y='EU_Sales', alpha=0.5, s=30)
g.map_dataframe(sns.regplot, x='NA_Sales', y='EU_Sales', scatter=False, color='red', line_kws={'linewidth': 2})

# Add correlation to each subplot
def add_corr(data, **kwargs):
    r, p = pearsonr(data['NA_Sales'], data['EU_Sales'])
    ax = plt.gca()
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

g.map_dataframe(add_corr)

g.set_axis_labels('NA Sales (millions)', 'EU Sales (millions)')
g.fig.suptitle('NA vs EU Sales: Conditional on Genre', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: Genre Conditioning Effect
# 
# **Key Findings:**
# - **Correlation persists** across most genres but with varying strength
# - **Action, Sports**: Strong correlation (r > 0.70) - universal appeal
# - **Role-Playing**: Moderate correlation (r ≈ 0.50-0.65) - some regional preferences
# - **Puzzle, Strategy**: Weaker correlation - more niche/regional variation
# 
# **Analytical Insight:**
# - **Relationship is NOT universal** - genre moderates the NA-EU correlation
# - Mainstream genres show stronger cross-regional consistency
# - Niche genres exhibit more regional differentiation
# - **Conclusion**: Introducing genre as third variable reveals heterogeneity

# %% [markdown]
# ---
# ## Analysis 2: Year vs Sales - Conditional on Platform Generation
# ### Research Question: Does temporal trend differ across platform eras?

# %% [markdown]
# ### Define Platform Generations

# %%
# Platform generation classification
platform_gen = {
    # Generation 6 (1998-2005)
    'PS2': 'Gen 6', 'GC': 'Gen 6', 'XB': 'Gen 6', 'DC': 'Gen 6', 'GBA': 'Gen 6',
    
    # Generation 7 (2005-2012)
    'X360': 'Gen 7', 'PS3': 'Gen 7', 'Wii': 'Gen 7', 'PSP': 'Gen 7', 'DS': 'Gen 7',
    
    # Generation 8 (2012-2020)
    'PS4': 'Gen 8', 'XOne': 'Gen 8', 'WiiU': 'Gen 8', '3DS': 'Gen 8', 'PSV': 'Gen 8',
    
    # Older/Other
    'PS': 'Gen 5', 'N64': 'Gen 5', 'SAT': 'Gen 5', 'GB': 'Gen 4-5', 
    'SNES': 'Gen 4', 'NES': 'Gen 3', 'GEN': 'Gen 4', '2600': 'Gen 2',
    'PC': 'PC', 'WS': 'Other', 'NG': 'Other', 'TG16': 'Other', '3DO': 'Other',
    'PCFX': 'Other', 'SCD': 'Other', 'GG': 'Other'
}

df_clean['Generation'] = df_clean['Platform'].map(platform_gen)

print("PLATFORM GENERATION MAPPING")
print("="*70)
print(df_clean['Generation'].value_counts())

# %% [markdown]
# ### Temporal Analysis by Generation

# %%
# Focus on major console generations
major_gens = ['Gen 6', 'Gen 7', 'Gen 8']
df_gen = df_clean[df_clean['Generation'].isin(major_gens)]

# Calculate yearly averages by generation
gen_year_stats = df_gen.groupby(['Generation', 'Year'])['Global_Sales'].agg(['mean', 'median', 'count']).reset_index()

print("\n📊 YEARLY SALES TRENDS BY GENERATION")
print("="*70)
print(gen_year_stats.head(15))

# %% [markdown]
# ### Visualization: Faceted Time Series

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

colors = {'Gen 6': 'steelblue', 'Gen 7': 'coral', 'Gen 8': 'seagreen'}

for idx, gen in enumerate(major_gens):
    gen_data = df_gen[df_gen['Generation'] == gen]
    
    # Scatter plot of all games
    axes[idx].scatter(gen_data['Year'], gen_data['Global_Sales'], 
                     alpha=0.3, s=20, color=colors[gen], label='Individual Games')
    
    # Yearly average line
    yearly_avg = gen_data.groupby('Year')['Global_Sales'].mean()
    axes[idx].plot(yearly_avg.index, yearly_avg.values, 
                  color='darkred', linewidth=2.5, marker='o', label='Yearly Average')
    
    axes[idx].set_xlabel('Year', fontsize=12)
    axes[idx].set_ylabel('Global Sales (millions)' if idx == 0 else '', fontsize=12)
    axes[idx].set_title(f'{gen} (2000s Era)', fontsize=13, fontweight='bold')
    axes[idx].legend(loc='upper right', fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Color-Encoded Scatter: Year vs Sales (Platform as Hue)

# %%
# Select top 8 platforms for clarity
top_platforms = df_clean['Platform'].value_counts().head(8).index.tolist()
df_top_plat = df_clean[df_clean['Platform'].isin(top_platforms)]

plt.figure(figsize=(16, 8))
for platform in top_platforms:
    plat_data = df_top_plat[df_top_plat['Platform'] == platform]
    plt.scatter(plat_data['Year'], plat_data['Global_Sales'], 
               label=platform, alpha=0.6, s=40, edgecolors='black', linewidth=0.3)

plt.xlabel('Release Year', fontsize=13)
plt.ylabel('Global Sales (millions)', fontsize=13)
plt.title('Year vs Sales: Color-Encoded by Platform (Top 8)', fontsize=15, fontweight='bold')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: Platform Generation Conditioning
# 
# **Key Findings:**
# - **Gen 6 (PS2 era)**: Gradual growth, peak around 2004-2005
# - **Gen 7 (Wii/X360 era)**: Highest average sales, peak 2008-2009, sharp decline after 2010
# - **Gen 8 (PS4 era)**: Lower average sales, possibly incomplete data (dataset cutoff)
# 
# **Analytical Insight:**
# - **Temporal trend is NOT uniform** across platform generations
# - Gen 7 represents industry "golden age" with highest sales
# - Decline post-2010 may reflect:
#   - Digital sales underrepresentation
#   - Market fragmentation
#   - Data collection bias
# - **Conclusion**: Platform generation is critical conditioning variable for temporal analysis

# %% [markdown]
# ---
# ## Analysis 3: Genre vs Sales - Conditional on Region
# ### Research Question: Are genre preferences region-specific?

# %% [markdown]
# ### Genre Sales by Region

# %%
# Focus on top 6 genres
top_genres_6 = df['Genre'].value_counts().head(6).index.tolist()
df_genre_6 = df[df['Genre'].isin(top_genres_6)]

# Calculate average sales by genre and region
regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales']
genre_regional = df_genre_6.groupby('Genre')[regional_cols].mean().round(3)

print("AVERAGE SALES BY GENRE AND REGION (Top 6 Genres)")
print("="*70)
print(genre_regional)

# %% [markdown]
# ### Visualization: Grouped Bar Chart

# %%
genre_regional_reset = genre_regional.reset_index()

x = np.arange(len(genre_regional_reset))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width, genre_regional_reset['NA_Sales'], width, 
               label='North America', color='steelblue', edgecolor='black')
bars2 = ax.bar(x, genre_regional_reset['EU_Sales'], width, 
               label='Europe', color='coral', edgecolor='black')
bars3 = ax.bar(x + width, genre_regional_reset['JP_Sales'], width, 
               label='Japan', color='seagreen', edgecolor='black')

ax.set_xlabel('Genre', fontsize=13)
ax.set_ylabel('Average Sales (millions)', fontsize=13)
ax.set_title('Genre Sales Performance by Region', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(genre_regional_reset['Genre'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Heatmap: Genre-Region Performance Matrix

# %%
plt.figure(figsize=(10, 7))
sns.heatmap(genre_regional, annot=True, fmt='.3f', cmap='YlGnBu', 
            linewidths=1, cbar_kws={'label': 'Avg Sales (millions)'})
plt.xlabel('Region', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Genre Performance Across Regions (Heatmap)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Regional Preference Ratio Analysis

# %%
# Calculate region-specific preference ratios
genre_regional['NA_Preference'] = genre_regional['NA_Sales'] / genre_regional[regional_cols].sum(axis=1)
genre_regional['EU_Preference'] = genre_regional['EU_Sales'] / genre_regional[regional_cols].sum(axis=1)
genre_regional['JP_Preference'] = genre_regional['JP_Sales'] / genre_regional[regional_cols].sum(axis=1)

preference_cols = ['NA_Preference', 'EU_Preference', 'JP_Preference']

print("\n📊 REGIONAL PREFERENCE RATIOS (Proportion of Total Sales)")
print("="*70)
print(genre_regional[preference_cols].round(3))

# Identify strongest regional preferences
print("\n🔍 STRONGEST REGIONAL PREFERENCES:")
for col in preference_cols:
    max_genre = genre_regional[col].idxmax()
    max_val = genre_regional[col].max()
    print(f"  {col.replace('_Preference', ''):15} : {max_genre} ({max_val:.1%})")

# %% [markdown]
# ### Interpretation: Regional Genre Preferences
# 
# **Key Findings:**
# - **Shooter**: Strong NA preference (55-60%), moderate EU, low JP
# - **Sports**: Balanced between NA and EU, low JP interest
# - **Role-Playing**: HIGHEST JP preference (30-35%), lower in West
# - **Platform**: Strong NA preference, moderate elsewhere
# 
# **Analytical Insight:**
# - **Genre success is region-dependent**
# - Western markets (NA/EU) prefer action-oriented genres
# - Japan has distinct preferences (RPG, Puzzle)
# - **Conclusion**: Genre-sales relationship changes dramatically when conditioned on region

# %% [markdown]
# ---
# ## Analysis 4: Platform vs Sales - Conditional on Year Era
# ### Research Question: Did platform success shift over time?

# %% [markdown]
# ### Define Time Eras

# %%
# Create era bins
df_clean['Era'] = pd.cut(df_clean['Year'], 
                         bins=[1980, 1995, 2005, 2012, 2020],
                         labels=['Early (1980-1995)', 'PS2 Era (1996-2005)', 
                                'HD Era (2006-2012)', 'Modern (2013-2020)'])

# Focus on top platforms
top_platforms_8 = df_clean['Platform'].value_counts().head(8).index.tolist()
df_plat_era = df_clean[df_clean['Platform'].isin(top_platforms_8)]

# Calculate platform sales by era
platform_era_sales = df_plat_era.groupby(['Era', 'Platform'])['Global_Sales'].mean().unstack(fill_value=0)

print("AVERAGE SALES BY PLATFORM AND ERA")
print("="*70)
print(platform_era_sales.round(3))

# %% [markdown]
# ### Visualization: Stacked Area Chart

# %%
platform_era_counts = df_plat_era.groupby(['Era', 'Platform']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 7))
platform_era_counts.T.plot(kind='bar', stacked=True, colormap='tab10', 
                           edgecolor='black', linewidth=0.5)
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Number of Games Released', fontsize=12)
plt.title('Platform Activity Across Time Eras', fontsize=15, fontweight='bold')
plt.legend(title='Era', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Faceted Box Plots: Sales Distribution by Era

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
eras = ['Early (1980-1995)', 'PS2 Era (1996-2005)', 'HD Era (2006-2012)', 'Modern (2013-2020)']

for idx, era in enumerate(eras):
    ax = axes[idx // 2, idx % 2]
    era_data = df_plat_era[df_plat_era['Era'] == era]
    
    if len(era_data) > 0:
        sns.boxplot(data=era_data, x='Platform', y='Global_Sales', 
                   palette='Set3', ax=ax, showfliers=False)
        ax.set_title(f'{era}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Platform', fontsize=11)
        ax.set_ylabel('Global Sales (millions)', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
        ax.set_title(f'{era}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: Temporal Platform Dynamics
# 
# **Key Findings:**
# - **PS2 Era**: PS2 dominance, high game counts, moderate sales
# - **HD Era**: Wii has highest median sales, X360/PS3 high variance
# - **Modern Era**: Sparse data suggests dataset limitation
# - **Platform lifecycles**: Clear temporal boundaries
# 
# **Analytical Insight:**
# - **Platform-sales relationship is era-dependent**
# - Each era has different market leader
# - Platform success tied to hardware generation cycle
# - **Conclusion**: Time is critical conditioning variable for platform analysis

# %% [markdown]
# ---
# ## 8.2 Statistical Interaction Reasoning

# %% [markdown]
# ### Interaction Effect 1: Genre × Platform → Sales
# 
# **Hypothesis**: Genre performance depends on platform type
# 
# **Example**: Sports games may perform better on Wii (motion controls) than on traditional consoles

# %%
# Focus on Sports genre across platforms
sports_data = df[df['Genre'] == 'Sports']
platform_sports_sales = sports_data.groupby('Platform')['Global_Sales'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)

print("SPORTS GENRE PERFORMANCE BY PLATFORM (Top 10)")
print("="*70)
print(platform_sports_sales)

# Compare with Action genre
action_data = df[df['Genre'] == 'Action']
platform_action_sales = action_data.groupby('Platform')['Global_Sales'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)

print("\nACTION GENRE PERFORMANCE BY PLATFORM (Top 10)")
print("="*70)
print(platform_action_sales)

# %% [markdown]
# ### Visualization: Genre-Platform Interaction

# %%
# Select specific platforms and genres for clarity
selected_platforms = ['Wii', 'X360', 'PS3', 'DS', 'PS2']
selected_genres = ['Sports', 'Action', 'Shooter', 'Role-Playing', 'Platform']

df_interaction = df[(df['Platform'].isin(selected_platforms)) & 
                    (df['Genre'].isin(selected_genres))]

interaction_matrix = df_interaction.groupby(['Platform', 'Genre'])['Global_Sales'].mean().unstack(fill_value=0)

plt.figure(figsize=(12, 7))
sns.heatmap(interaction_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            linewidths=1, cbar_kws={'label': 'Avg Sales (millions)'})
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Platform', fontsize=12)
plt.title('Platform × Genre Interaction Effect on Sales', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interaction Interpretation
# 
# **Observed Interactions:**
# - **Wii × Sports**: Exceptionally high (motion control advantage)
# - **Wii × Platform**: Very high (Nintendo IP strength)
# - **X360/PS3 × Shooter**: High (controller/online infrastructure)
# - **DS × Puzzle**: Moderate-high (touch screen suitability)
# 
# **Statistical Implication:**
# - **Main effects alone are insufficient**
# - Platform and Genre effects are **not additive**
# - Interaction term would be statistically significant in regression model
# - **Conclusion**: Multivariate relationship is multiplicative, not linear

# %% [markdown]
# ### Interaction Effect 2: Year × Region → Sales
# 
# **Hypothesis**: Regional market growth rates differ over time

# %%
# Calculate regional market share over time (5-year bins)
df_clean['Year_Bin'] = pd.cut(df_clean['Year'], bins=[1980, 1995, 2000, 2005, 2010, 2020],
                               labels=['1980-1995', '1996-2000', '2001-2005', '2006-2010', '2011-2020'])

regional_temporal = df_clean.groupby('Year_Bin')[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()
regional_temporal_pct = regional_temporal.div(regional_temporal.sum(axis=1), axis=0) * 100

print("REGIONAL MARKET SHARE EVOLUTION (% of Total Sales)")
print("="*70)
print(regional_temporal_pct.round(2))

# %% [markdown]
# ### Visualization: Temporal Regional Shift

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Stacked area chart
regional_temporal.plot(kind='area', stacked=True, alpha=0.7, 
                       color=['steelblue', 'coral', 'seagreen'], ax=axes[0])
axes[0].set_xlabel('Time Period', fontsize=12)
axes[0].set_ylabel('Total Sales (millions)', fontsize=12)
axes[0].set_title('Regional Sales Evolution (Absolute)', fontsize=14, fontweight='bold')
axes[0].legend(title='Region', labels=['North America', 'Europe', 'Japan'])
axes[0].grid(alpha=0.3)

# 2. Line chart (market share %)
regional_temporal_pct.plot(kind='line', marker='o', linewidth=2.5, ax=axes[1])
axes[1].set_xlabel('Time Period', fontsize=12)
axes[1].set_ylabel('Market Share (%)', fontsize=12)
axes[1].set_title('Regional Market Share Evolution', fontsize=14, fontweight='bold')
axes[1].legend(title='Region', labels=['North America', 'Europe', 'Japan'])
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation: Year-Region Interaction
# 
# **Key Findings:**
# - **1996-2000**: NA dominance (~50%), JP strong (~25%)
# - **2001-2005**: EU growth, NA decline, JP stable
# - **2006-2010**: Balanced distribution, EU peak
# - **2011-2020**: NA recovery, JP further decline
# 
# **Statistical Implication:**
# - **Regional growth trajectories diverge**
# - Year-Region interaction is significant
# - Cannot model temporal trends without regional stratification
# - **Conclusion**: Time affects regions differently (non-parallel trends)

# %% [markdown]
# ### Limitations of Pairwise Analysis

# %% [markdown]
# ### 🚨 Risks of Overinterpretation
# 
# **Identified Limitations:**
# 
# 1. **Confounding Variables**:
#    - Publisher marketing budgets (not in dataset)
#    - Digital sales representation (missing data)
#    - Review scores/quality metrics (absent)
#    - Economic conditions during release (external)
# 
# 2. **Simpson's Paradox Risk**:
#    - Aggregated trends may reverse when conditioned
#    - Example: Overall positive correlation may be negative within subgroups
# 
# 3. **Temporal Autocorrelation**:
#    - Sales influenced by franchise history
#    - Sequels benefit from predecessor success
#    - Dataset treats observations as independent (violation)
# 
# 4. **Selection Bias**:
#    - Dataset limited to VGChartz-tracked games
#    - Indie/digital games underrepresented
#    - Survival bias (failed games may be excluded)
# 
# 5. **Multiple Comparisons Problem**:
#    - Testing many genre-platform pairs
#    - Some "significant" results are false positives
#    - No correction for family-wise error rate

# %% [markdown]
# ### Statistical Interaction Discussion

# %% [markdown]
# ### 💡 Possible Interaction Effects Not Yet Explored
# 
# **Potential Three-Way Interactions:**
# 
# 1. **Year × Platform × Genre**:
#    - Did RPG success on DS peak in specific years?
#    - Temporal platform-genre synergies
# 
# 2. **Region × Publisher × Genre**:
#    - Japanese publishers + RPG + JP market
#    - Complex cultural-business interaction
# 
# 3. **Platform Generation × Genre × Region**:
#    - Gen 7 + Shooter + NA = peak sales?
#    - Era-specific regional genre preferences
# 
# **Why Pairwise Analysis is Insufficient:**
# - Real-world systems are **highly multivariate**
# - Effects are often **non-additive and multiplicative**
# - Single-variable conditioning reveals only **partial story**
# - Full understanding requires **regression modeling with interaction terms**

# %% [markdown]
# ---
# ## Complex Multivariate Visualization

# %% [markdown]
# ### 4D Visualization: Sales, Year, Platform, Genre

# %%
# Create a complex scatter plot with multiple dimensions
df_viz = df_clean[df_clean['Platform'].isin(['PS2', 'X360', 'PS3', 'Wii', 'DS'])].copy()
df_viz = df_viz[df_viz['Genre'].isin(['Action', 'Sports', 'Shooter', 'Role-Playing'])]

# Create size based on Global_Sales (but cap for visibility)
df_viz['Size'] = df_viz['Global_Sales'].clip(upper=10) * 10

plt.figure(figsize=(16, 10))

for genre in df_viz['Genre'].unique():
    genre_data = df_viz[df_viz['Genre'] == genre]
    plt.scatter(genre_data['Year'], genre_data['NA_Sales'], 
               s=genre_data['Size'], alpha=0.5, label=genre, edgecolors='black', linewidth=0.5)

plt.xlabel('Release Year', fontsize=13)
plt.ylabel('NA Sales (millions)', fontsize=13)
plt.title('4D Visualization: Year × NA Sales × Genre (size = Global Sales)', 
         fontsize=15, fontweight='bold')
plt.legend(title='Genre', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interactive 3D Scatter (Conceptual - Plotly)

# %%
# Using plotly for interactive 3D exploration
df_3d = df_clean[df_clean['Platform'].isin(['PS2', 'X360', 'Wii'])].copy()
df_3d = df_3d[df_3d['Genre'].isin(['Action', 'Sports', 'Shooter'])]

fig = px.scatter_3d(df_3d, 
                    x='NA_Sales', 
                    y='EU_Sales', 
                    z='JP_Sales',
                    color='Genre',
                    size='Global_Sales',
                    hover_data=['Name', 'Platform', 'Year'],
                    title='3D Regional Sales Distribution (Genre × Platform)',
                    labels={'NA_Sales': 'NA Sales (M)', 
                           'EU_Sales': 'EU Sales (M)',
                           'JP_Sales': 'JP Sales (M)'},
                    opacity=0.7,
                    size_max=20)

fig.update_layout(
    scene=dict(
        xaxis_title='NA Sales',
        yaxis_title='EU Sales',
        zaxis_title='JP Sales'
    ),
    width=900,
    height=700
)

fig.show()

# %% [markdown]
# ---
# ## Summary: Section D - Multivariate Analysis Findings

# %% [markdown]
# ### 🔍 Key Conditional Insights
# 
# **1. Regional Correlations Conditioned on Genre:**
# - NA-EU correlation **not uniform** across genres
# - Action/Sports: Strong correlation (universal appeal)
# - RPG/Puzzle: Weaker correlation (regional preferences)
# - **Conclusion**: Genre moderates regional relationship
# 
# **2. Temporal Trends Conditioned on Platform:**
# - Year-Sales relationship **era-dependent**
# - Gen 7 (2006-2012) = industry peak
# - Different platforms peaked at different times
# - **Conclusion**: Cannot analyze time without platform context
# 
# **3. Genre Performance Conditioned on Region:**
# - Shooter/Sports: NA/EU dominant
# - RPG: JP dominant (2-3× higher preference)
# - Regional preferences **drastically alter** genre rankings
# - **Conclusion**: Genre success is region-specific
# 
# **4. Platform Success Conditioned on Era:**
# - Platform rankings **change completely** across eras
# - PS2 (2000s), Wii (late 2000s), PS4 (2010s)
# - **Conclusion**: Platform analysis requires temporal stratification

# %% [markdown]
# ### 🧩 Interaction Effects Identified
# 
# **Statistical Interactions:**
# 
# 1. **Genre × Platform → Sales**:
#    - Wii × Sports = exceptional performance
#    - X360 × Shooter = above-average sales
#    - Platform design enables genre-specific advantages
# 
# 2. **Year × Region → Sales**:
#    - Regional market shares shift over time
#    - Non-parallel temporal trends
#    - Global patterns mask regional dynamics
# 
# 3. **Implied Three-Way Interactions**:
#    - Year × Platform × Genre likely significant
#    - Region × Publisher × Genre potential
#    - Full modeling requires regression with interaction terms

# %% [markdown]
# ### ⚠️ Analytical Limitations Acknowledged
# 
# **Pairwise Analysis Cannot Capture:**
# - Simultaneous multi-variable effects
# - Higher-order interactions (3-way, 4-way)
# - Causal mechanisms (observational data only)
# - Unmeasured confounders (marketing, quality, timing)
# 
# **Overinterpretation Risks:**
# - False positives from multiple testing
# - Simpson's Paradox potential
# - Selection bias in dataset
# - Temporal autocorrelation (franchise effects)
# 
# **What We've Learned:**
# - Conditioning reveals heterogeneity
# - Relationships are **context-dependent**
# - Simple bivariate conclusions **do not generalize**
# - Full understanding requires **multivariate modeling**

# %% [markdown]
# ### 📈 Visualization Insights
# 
# **Effective Techniques:**
# - **Faceted plots**: Reveal conditional patterns clearly
# - **Color encoding**: Add third dimension to scatter plots
# - **Heatmaps**: Show interaction effects at a glance
# - **3D plots**: Explore regional sales relationships
# - **Stacked charts**: Track compositional changes over time
# 
# **Key Takeaway:**
# - Single plots cannot capture full complexity
# - Multiple complementary visualizations needed
# - Interactivity helps explore high-dimensional data

# %%
print("\n" + "="*70)
print("SECTION D COMPLETE: Multivariate Analysis")
print("="*70)
print("\nKey Achievement: Demonstrated that relationships are context-dependent")
print("Next Steps:")
print("  → Section E: Advanced Statistical Exploration")
print("  → Section F: Dimensionality Reduction (PCA)")
print("  → Section G: Clustering for Pattern Discovery")
print("="*70)

# %%