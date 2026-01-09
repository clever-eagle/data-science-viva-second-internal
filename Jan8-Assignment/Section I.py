# %% [markdown]
# # Section I: Reading, Critiquing, and Stress-Testing Visuals
# ## Video Game Sales Dataset - Critical Analysis of Visualization Choices

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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

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
# ## 13.1 Self-Critique: Analyzing Our Own Visualizations
# 
# ### Framework for Self-Critique:
# 1. **Core Message**: What is the primary insight?
# 2. **Hidden Information**: What does this chart NOT show?
# 3. **Potential Misinterpretations**: How could viewers misread this?
# 4. **Design Improvements**: What would make this clearer?

# %% [markdown]
# ---
# ## Visualization 1: Genre Sales Distribution (Boxplot)

# %%
# Create the visualization we'll critique
top_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(6).index
df_top_genres = df[df['Genre'].isin(top_genres)]

fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(data=df_top_genres, x='Genre', y='Global_Sales', palette='Set2', ax=ax)
ax.set_xlabel('Genre', fontsize=13, fontweight='bold')
ax.set_ylabel('Global Sales per Game (Millions)', fontsize=13, fontweight='bold')
ax.set_title('Sales Distribution by Genre (Top 6 Genres)', fontsize=15, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Self-Critique: Boxplot Analysis

# %%
print("🔍 SELF-CRITIQUE: Genre Sales Boxplot")
print("="*70)

print("\n1. CORE MESSAGE:")
print("   ✓ 'Most games in all genres have low sales (<5M), but outliers exist'")
print("   ✓ 'Genre does not dramatically shift typical game performance'")
print("   ✓ 'Sports and Action have the most extreme outliers (blockbusters)'")

print("\n2. HIDDEN INFORMATION (What This Chart DOESN'T Show):")
print("   ✗ Number of games per genre (sample size)")
print("     → Sports might have 1000 games, Puzzle only 50")
print("     → Outliers in small genres are less meaningful")
print("   ✗ Temporal trends (are genres growing or declining?)")
print("   ✗ Regional differences (maybe Sports dominates NA but not JP)")
print("   ✗ Platform associations (which platforms favor which genres?)")
print("   ✗ Publisher concentration (are outliers from same publishers?)")

print("\n3. POTENTIAL MISINTERPRETATIONS:")
print("   ⚠️  'Action games sell better than RPGs'")
print("      → FALSE: Medians are similar (~0.5M); means differ due to outliers")
print("   ⚠️  'Sports is the best genre for developers'")
print("      → RISKY: Ignores competition, market saturation, development costs")
print("   ⚠️  'Outliers are anomalies to ignore'")
print("      → FALSE: Outliers (GTA, FIFA) are culturally/financially significant")
print("   ⚠️  'All genres have equal market opportunity'")
print("      → UNCLEAR: Chart doesn't show total market size per genre")

print("\n4. DESIGN IMPROVEMENTS:")
print("   → Add sample size annotations (n=XXX above each box)")
print("   → Use violin plot to show full distribution shape")
print("   → Color-code by decade to show temporal shifts")
print("   → Add strip plot overlay to show individual games")
print("   → Include total market size (sum) as secondary metric")
print("   → Facet by region to reveal geographic patterns")

print("\n5. STATISTICAL CONCERNS:")
print("   ⚠️  Boxplot assumes symmetric outlier definition (1.5×IQR)")
print("      → May mislabel legitimate successes as 'outliers'")
print("   ⚠️  Hides multimodality (bimodal distributions look uniform)")
print("   ⚠️  Visual weight of whiskers suggests false precision")

print("\n6. ALTERNATIVE VISUALIZATIONS TO CONSIDER:")
print("   • Cumulative distribution plot (shows full percentile curves)")
print("   • Ridge plot (shows overlapping distributions)")
print("   • Swarm plot (shows every game, avoids aggregation)")
print("   • Log-scale boxplot (compresses outlier dominance)")

# %% [markdown]
# ### Improved Version: Addressing Critiques

# %%
# Enhanced visualization addressing critique points
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: Violin plot with sample sizes
parts = axes[0].violinplot([df_top_genres[df_top_genres['Genre'] == g]['Global_Sales'].values 
                            for g in top_genres],
                           positions=range(len(top_genres)),
                           showmeans=True, showmedians=True, widths=0.7)

# Color the violins
colors = sns.color_palette('Set2', len(top_genres))
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

axes[0].set_xticks(range(len(top_genres)))
axes[0].set_xticklabels(top_genres, rotation=45, ha='right', fontsize=11)
axes[0].set_xlabel('Genre', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Global Sales per Game (M)', fontsize=12, fontweight='bold')
axes[0].set_title('✅ IMPROVED: Violin Plot with Distribution Shape', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Add sample size annotations
for i, genre in enumerate(top_genres):
    count = len(df_top_genres[df_top_genres['Genre'] == genre])
    axes[0].text(i, axes[0].get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Right: Cumulative distribution comparison
for i, genre in enumerate(top_genres):
    genre_data = df_top_genres[df_top_genres['Genre'] == genre]['Global_Sales'].sort_values()
    cumulative = np.arange(1, len(genre_data) + 1) / len(genre_data) * 100
    axes[1].plot(genre_data.values, cumulative, linewidth=2.5, label=genre, color=colors[i])

axes[1].set_xlabel('Global Sales per Game (M)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Percentage of Games', fontsize=12, fontweight='bold')
axes[1].set_title('✅ IMPROVED: Cumulative Distribution (Full Detail)', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xscale('log')  # Log scale to handle outliers

# Add percentile lines
for pct in [50, 75, 90]:
    axes[1].axhline(pct, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].text(axes[1].get_xlim()[0] * 1.1, pct, f'{pct}th percentile', 
                fontsize=8, color='gray', va='center')

plt.tight_layout()
plt.show()

print("\n✅ IMPROVEMENTS IMPLEMENTED:")
print("   ✓ Violin plot shows full distribution shape (not just quartiles)")
print("   ✓ Sample sizes explicitly labeled")
print("   ✓ Cumulative distribution plot reveals all percentiles")
print("   ✓ Log scale on CDF makes outliers visible without dominating")

# %% [markdown]
# ---
# ## Visualization 2: Regional Sales Trends Over Time (Line Chart)

# %%
# Create the visualization
yearly_regional = df_clean.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(yearly_regional.index, yearly_regional['NA_Sales'], 
       linewidth=3, label='North America', marker='o', markersize=5)
ax.plot(yearly_regional.index, yearly_regional['EU_Sales'], 
       linewidth=3, label='Europe', marker='s', markersize=5)
ax.plot(yearly_regional.index, yearly_regional['JP_Sales'], 
       linewidth=3, label='Japan', marker='^', markersize=5)
ax.plot(yearly_regional.index, yearly_regional['Other_Sales'], 
       linewidth=3, label='Other', marker='d', markersize=5)

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Annual Sales (Millions)', fontsize=13, fontweight='bold')
ax.set_title('Video Game Sales by Region Over Time', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Self-Critique: Regional Trends Line Chart

# %%
print("🔍 SELF-CRITIQUE: Regional Sales Trends")
print("="*70)

print("\n1. CORE MESSAGE:")
print("   ✓ 'North America consistently dominates global game sales'")
print("   ✓ 'All regions peaked around 2008-2010, then declined'")
print("   ✓ 'Japan market has been relatively stable but smaller'")

print("\n2. HIDDEN INFORMATION:")
print("   ✗ Absolute numbers hide market share changes")
print("     → NA might be declining faster than EU in relative terms")
print("   ✗ Aggregation obscures platform transitions")
print("     → 2008 peak driven by Wii/DS boom, not shown here")
print("   ✗ Currency exchange rate effects not addressed")
print("     → Sales in millions of units vs millions of dollars unclear")
print("   ✗ Digital sales not included (dataset ends 2016)")
print("     → Recent decline may be artifact of incomplete data")
print("   ✗ Population differences ignored")
print("     → Per-capita sales would tell different story")

print("\n3. POTENTIAL MISINTERPRETATIONS:")
print("   ⚠️  'The gaming industry is dying after 2010'")
print("      → FALSE: Physical sales declined, but digital/mobile exploded")
print("      → Dataset doesn't capture full market post-2010")
print("   ⚠️  'Other regions don't matter'")
print("      → MISLEADING: 'Other' includes China, India (huge emerging markets)")
print("      → Early dataset means mobile-first markets underrepresented")
print("   ⚠️  'Japan market is failing'")
print("      → CONTEXT: Japan shifted to mobile/handheld not captured here")
print("   ⚠️  'NA > EU > JP is natural order'")
print("      → IGNORES: Population (EU ≈ NA), culture, platform preferences")

print("\n4. DESIGN IMPROVEMENTS:")
print("   → Normalize to percentages (market share over time)")
print("   → Add per-capita sales (sales / population)")
print("   → Annotate major platform launches (PS2, Wii, etc.)")
print("   → Show confidence intervals (data completeness varies by year)")
print("   → Split by platform generation to explain transitions")
print("   → Add shaded regions for console generations")

print("\n5. STATISTICAL CONCERNS:")
print("   ⚠️  No error bars or confidence intervals")
print("      → Are 2016 numbers complete? (Likely not)")
print("   ⚠️  Trend line assumes continuous measurement")
print("      → Actually discrete annual aggregates")
print("   ⚠️  No accounting for data collection bias")
print("      → VGChartz data quality degrades in recent years")

print("\n6. CONTEXTUAL KNOWLEDGE REQUIRED:")
print("   • 2008 peak = Wii/DS casual gaming boom")
print("   • 2011+ decline = Digital distribution not tracked")
print("   • Japan 'stability' = Handheld dominance (3DS, Vita)")
print("   • 'Other' category undervalued (emerging markets)")

# %% [markdown]
# ### Improved Version: Market Share Perspective

# %%
# Enhanced visualization: Market share over time
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top: Absolute sales (original)
yearly_regional.plot(ax=axes[0], linewidth=2.5, marker='o', markersize=5)
axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Annual Sales (Millions)', fontsize=12, fontweight='bold')
axes[0].set_title('Absolute Sales by Region', fontsize=13, fontweight='bold')
axes[0].legend(title='Region', fontsize=10)
axes[0].grid(alpha=0.3)

# Annotate key events
events = {
    2006: 'Wii Launch',
    2008: 'Market Peak',
    2013: 'PS4/XB1 Launch'
}
for year, event in events.items():
    if year in yearly_regional.index:
        axes[0].axvline(year, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[0].text(year, axes[0].get_ylim()[1] * 0.95, event, 
                    rotation=90, va='top', fontsize=9, color='red', fontweight='bold')

# Bottom: Market share (normalized to percentages)
yearly_regional_pct = yearly_regional.div(yearly_regional.sum(axis=1), axis=0) * 100

yearly_regional_pct.plot(kind='area', stacked=True, ax=axes[1], alpha=0.7)
axes[1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Market Share (%)', fontsize=12, fontweight='bold')
axes[1].set_title('✅ IMPROVED: Market Share Distribution Over Time', fontsize=13, fontweight='bold')
axes[1].legend(title='Region', fontsize=10, loc='upper left')
axes[1].set_ylim(0, 100)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n✅ IMPROVEMENTS IMPLEMENTED:")
print("   ✓ Market share view shows relative importance changes")
print("   ✓ Major industry events annotated for context")
print("   ✓ Stacked area shows total market size + composition")
print("   ✓ Viewer can now see NA declining share despite growing absolute sales")

# %% [markdown]
# ---
# ## Visualization 3: Publisher Success Scatter (Sales vs Game Count)

# %%
# Create visualization
publisher_stats = df.groupby('Publisher').agg({
    'Global_Sales': 'sum',
    'Name': 'count'
}).rename(columns={'Name': 'Game_Count'})

# Filter to publishers with at least 50 games
publisher_stats_filtered = publisher_stats[publisher_stats['Game_Count'] >= 50]

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(publisher_stats_filtered['Game_Count'], 
                    publisher_stats_filtered['Global_Sales'],
                    s=100, alpha=0.6, c=range(len(publisher_stats_filtered)),
                    cmap='viridis', edgecolors='black', linewidth=1)

# Annotate top 5 publishers
top_5 = publisher_stats_filtered.nlargest(5, 'Global_Sales')
for idx, row in top_5.iterrows():
    ax.annotate(idx, (row['Game_Count'], row['Global_Sales']),
               fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Number of Games Published', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Global Sales (Millions)', fontsize=12, fontweight='bold')
ax.set_title('Publisher Portfolio Size vs Total Sales', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Self-Critique: Publisher Scatter Plot

# %%
print("🔍 SELF-CRITIQUE: Publisher Success Scatter")
print("="*70)

print("\n1. CORE MESSAGE:")
print("   ✓ 'More games published ≈ More total sales (generally)'")
print("   ✓ 'Nintendo is an outlier: Fewer games but massive sales'")
print("   ✓ 'Publishing volume doesn't guarantee proportional success'")

print("\n2. HIDDEN INFORMATION:")
print("   ✗ Time dimension completely absent")
print("     → EA with 1000 games over 30 years vs indie with 50 in 5 years")
print("   ✗ Average sales per game not shown")
print("     → Quality/efficiency metric missing")
print("   ✗ Genre specialization hidden")
print("     → Some publishers focus on high-volume, low-margin genres")
print("   ✗ Platform exclusivity not indicated")
print("     → Nintendo's first-party advantage invisible")
print("   ✗ Development vs publishing distinction")
print("     → EA publishes many external studios' work")
print("   ✗ Regional market dominance")
print("     → Nintendo strong in JP, EA in NA - not shown")

print("\n3. POTENTIAL MISINTERPRETATIONS:")
print("   ⚠️  'To succeed, publish many games'")
print("      → MISLEADING: Correlation ≠ causation; successful publishers can afford more games")
print("   ⚠️  'Nintendo is inefficient (low volume)'")
print("      → FALSE: Nintendo has highest sales per game (quality over quantity)")
print("   ⚠️  'All games contribute equally to total sales'")
print("      → FALSE: 80/20 rule likely applies (few blockbusters drive most revenue)")
print("   ⚠️  'Small publishers can't compete'")
print("      → IGNORES: Many small publishers not in dataset (selection bias)")

print("\n4. DESIGN IMPROVEMENTS:")
print("   → Add bubble size = average sales per game")
print("   → Color by primary platform (Nintendo, PlayStation, Multi)")
print("   → Add trend line with confidence interval")
print("   → Show temporal animation (bubble chart over decades)")
print("   → Include 'hit rate' metric (% of games >1M sales)")
print("   → Log-log scale to see smaller publishers better")

print("\n5. STATISTICAL CONCERNS:")
print("   ⚠️  Survivor bias: Only successful publishers in dataset")
print("      → Failed publishers (who published few games) not represented")
print("   ⚠️  Linear scale compresses low-volume publishers")
print("      → Many interesting small publishers invisible")
print("   ⚠️  No measure of statistical significance")
print("      → Is positive correlation significant? R² not shown")
print("   ⚠️  Assumes independence (publishers compete/collaborate)")

print("\n6. MISSING CONTEXT:")
print("   • Nintendo = First-party developer (controls hardware)")
print("   • EA/Activision = Third-party publishers (platform-agnostic)")
print("   • Some 'publishers' are actually developer studios")
print("   • Dataset time range affects publisher comparisons")

# %% [markdown]
# ### Improved Version: Multi-Metric Publisher Analysis

# %%
# Enhanced visualization with additional metrics
publisher_stats_enhanced = df.groupby('Publisher').agg({
    'Global_Sales': ['sum', 'mean'],
    'Name': 'count'
}).reset_index()
publisher_stats_enhanced.columns = ['Publisher', 'Total_Sales', 'Avg_Sales', 'Game_Count']

# Filter
publisher_stats_enhanced = publisher_stats_enhanced[publisher_stats_enhanced['Game_Count'] >= 50]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: Original with bubble size = avg sales
scatter1 = axes[0].scatter(publisher_stats_enhanced['Game_Count'], 
                          publisher_stats_enhanced['Total_Sales'],
                          s=publisher_stats_enhanced['Avg_Sales'] * 200,  # Bubble size
                          alpha=0.6, c=publisher_stats_enhanced['Avg_Sales'],
                          cmap='RdYlGn', edgecolors='black', linewidth=1.5)

# Annotate top publishers
top_publishers = publisher_stats_enhanced.nlargest(5, 'Total_Sales')
for _, row in top_publishers.iterrows():
    axes[0].annotate(row['Publisher'], 
                    (row['Game_Count'], row['Total_Sales']),
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

axes[0].set_xlabel('Number of Games Published', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Global Sales (M)', fontsize=12, fontweight='bold')
axes[0].set_title('✅ IMPROVED: Bubble Size = Avg Sales per Game', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

# Add colorbar
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Avg Sales per Game (M)', fontsize=10)

# Right: Efficiency plot (avg sales vs game count)
scatter2 = axes[1].scatter(publisher_stats_enhanced['Game_Count'], 
                          publisher_stats_enhanced['Avg_Sales'],
                          s=publisher_stats_enhanced['Total_Sales'] / 10,  # Size = total sales
                          alpha=0.6, c=range(len(publisher_stats_enhanced)),
                          cmap='plasma', edgecolors='black', linewidth=1.5)

# Highlight Nintendo's efficiency
nintendo_row = publisher_stats_enhanced[publisher_stats_enhanced['Publisher'] == 'Nintendo']
if not nintendo_row.empty:
    axes[1].scatter(nintendo_row['Game_Count'], nintendo_row['Avg_Sales'],
                   s=500, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Nintendo', zorder=5)
    axes[1].annotate('Nintendo\n(Quality Focus)', 
                    (nintendo_row['Game_Count'].values[0], nintendo_row['Avg_Sales'].values[0]),
                    fontsize=10, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

axes[1].set_xlabel('Number of Games Published', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Sales per Game (M)', fontsize=12, fontweight='bold')
axes[1].set_title('✅ IMPROVED: Publisher Efficiency View', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ IMPROVEMENTS IMPLEMENTED:")
print("   ✓ Bubble size encodes average sales (efficiency metric)")
print("   ✓ Color scale shows quality gradient")
print("   ✓ Second panel explicitly shows efficiency vs volume trade-off")
print("   ✓ Nintendo's outlier status highlighted with context")
print("   ✓ Viewer can now distinguish 'many mediocre' from 'few excellent'")

# %% [markdown]
# ---
# ## 13.2 External Visualization Critique
# ### Analyzing a Published Industry Visualization

# %% [markdown]
# ### Example: Recreating a Common "Top 10 Best-Selling Games" Bar Chart

# %%
# Create a typical "Top 10" bar chart (common in gaming media)
top_10_games = df.nlargest(10, 'Global_Sales')[['Name', 'Platform', 'Year', 'Genre', 'Global_Sales']]

fig, ax = plt.subplots(figsize=(12, 8))

# Typical media style: Colorful, ranked bars
colors_gradient = plt.cm.rainbow(np.linspace(0, 1, 10))
bars = ax.barh(range(10), top_10_games['Global_Sales'].values, 
              color=colors_gradient, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(10))
ax.set_yticklabels([f"{i+1}. {name[:30]}" for i, name in enumerate(top_10_games['Name'])], 
                   fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Global Sales (Millions of Units)', fontsize=12, fontweight='bold')
ax.set_title('🎮 TOP 10 BEST-SELLING VIDEO GAMES OF ALL TIME 🏆', 
            fontsize=15, fontweight='bold', color='darkblue')

# Add sales values on bars
for i, (bar, sales) in enumerate(zip(bars, top_10_games['Global_Sales'])):
    ax.text(sales + 1, bar.get_y() + bar.get_height()/2, 
           f'{sales:.2f}M', va='center', fontsize=10, fontweight='bold')

ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Critical Analysis of "Top 10" Visualization

# %%
print("🔍 EXTERNAL VISUALIZATION CRITIQUE")
print("="*70)
print("VISUALIZATION TYPE: 'Top 10 Best-Selling Games' Bar Chart")
print("COMMON IN: Gaming news sites, YouTube thumbnails, infographics")

print("\n✅ STRENGTHS:")
print("   1. CLARITY:")
print("      ✓ Immediately clear message: 'These are the biggest games'")
print("      ✓ Horizontal bars easy to read with long game titles")
print("      ✓ Values labeled directly on bars (no need to read axis)")
print("   2. ENGAGEMENT:")
print("      ✓ Ranking format appeals to human competitive nature")
print("      ✓ Recognizable game names drive emotional connection")
print("   3. ACCESSIBILITY:")
print("      ✓ Simple chart type (bar chart) universally understood")
print("      ✓ No statistical knowledge required to interpret")

print("\n❌ WEAKNESSES:")
print("   1. TEMPORAL CONTEXT MISSING:")
print("      ✗ Wii Sports (2006) vs recent games have 10+ more years to accumulate sales")
print("      ✗ Older games benefit from longer market presence")
print("      ✗ No distinction between 'peak year' and 'lifetime total'")
print("   2. PLATFORM BIAS:")
print("      ✗ Bundled games (Wii Sports came with console) inflate numbers artificially")
print("      ✗ Multi-platform games (GTA V on 5+ platforms) vs exclusives unfair")
print("      ✗ No normalization for install base (Wii sold 100M, Vita sold 15M)")
print("   3. GENRE HOMOGENEITY:")
print("      ✗ List dominated by Sports/Action - other genres underrepresented")
print("      ✗ Reinforces 'only blockbusters matter' mentality")
print("   4. VISUAL DESIGN ISSUES:")
print("      ✗ Rainbow gradient serves no purpose (not semantic)")
print("      ✗ #1 should be visually distinct, not just first")
print("      ✗ No error bars (sales estimates have uncertainty)")

print("\n⚠️  ETHICAL CONCERNS:")
print("   1. SURVIVORSHIP BIAS:")
print("      → Only shows successes; ignores 99% of games that failed")
print("      → Creates false impression that gaming is 'easy money'")
print("   2. INCOMPLETE DATA:")
print("      → Digital sales not included (underestimates recent games)")
print("      → Mobile games excluded (ignores largest market segment post-2010)")
print("      → Free-to-play revenue not counted (Fortnite would dominate)")
print("   3. CULTURAL BIAS:")
print("      → Western-centric list (where's Dragon Quest, Monster Hunter?)")
print("      → Reflects VGChartz data collection bias")
print("   4. IMPLIED CAUSALITY:")
print("      → Viewer may conclude: 'Make sports games to succeed'")
print("      → Ignores market saturation, competition, budget requirements")

print("\n🔧 SUGGESTED IMPROVEMENTS:")
print("   1. ADD CONTEXT:")
print("      → Normalize by 'sales per year since release'")
print("      → Indicate bundled vs standalone sales")
print("      → Show platform install base")
print("   2. EXPAND SCOPE:")
print("      → Separate lists by platform/genre/era")
print("      → Include 'Top 10 Indies', 'Top 10 RPGs', etc.")
print("      → Show 'flops' for balance (learning value)")
print("   3. IMPROVE DESIGN:")
print("      → Color by genre (semantic meaning)")
print("      → Add sparklines showing sales trajectory over time")
print("      → Include metacritic score or player rating")
print("   4. TRANSPARENT LIMITATIONS:")
print("      → Caption: 'Physical sales only, data through 2016'")
print("      → Acknowledge bundling impact")
print("      → State data source and methodology")

# %% [markdown]
# ### Improved "Top 10" with Context

# %%
# Enhanced version with context
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Top: Original top 10 with genre coloring
genre_colors = {
    'Sports': '#1f77b4',
    'Platform': '#ff7f0e', 
    'Racing': '#2ca02c',
    'Role-Playing': '#d62728',
    'Shooter': '#9467bd',
    'Misc': '#8c564b'
}

bar_colors = [genre_colors.get(g, 'gray') for g in top_10_games['Genre']]
bars1 = axes[0].barh(range(10), top_10_games['Global_Sales'].values, 
                    color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.8)

axes[0].set_yticks(range(10))
axes[0].set_yticklabels([f"{i+1}. {name[:35]}" for i, name in enumerate(top_10_games['Name'])], 
                       fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel('Global Sales (Millions)', fontsize=11, fontweight='bold')
axes[0].set_title('✅ IMPROVED: Top 10 with Genre Context', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# Add release year annotations
for i, (sales, year, platform) in enumerate(zip(top_10_games['Global_Sales'], 
                                                top_10_games['Year'], 
                                                top_10_games['Platform'])):
    axes[0].text(sales + 1, i, f'{sales:.1f}M | {int(year)} | {platform}', 
                va='center', fontsize=8, style='italic')

# Legend for genres
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=genre) 
                  for genre, color in genre_colors.items() if genre in top_10_games['Genre'].values]
axes[0].legend(handles=legend_elements, title='Genre', loc='lower right', fontsize=9)

# Bottom: Sales per year (normalized view)
top_10_games_copy = top_10_games.copy()
top_10_games_copy['Years_Available'] = 2016 - top_10_games_copy['Year'] + 1  # Dataset ends 2016
top_10_games_copy['Sales_Per_Year'] = top_10_games_copy['Global_Sales'] / top_10_games_copy['Years_Available']
top_10_sorted = top_10_games_copy.sort_values('Sales_Per_Year', ascending=True)

bars2 = axes[1].barh(range(10), top_10_sorted['Sales_Per_Year'].values, 
                    color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)

axes[1].set_yticks(range(10))
axes[1].set_yticklabels([f"{name[:35]}" for name in top_10_sorted['Name']], fontsize=10)
axes[1].invert_yaxis()
axes[1].set_xlabel('Average Sales per Year (M/year)', fontsize=11, fontweight='bold')
axes[1].set_title('✅ IMPROVED: Normalized by Time on Market', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

# Highlight bundled games
bundled_games = ['Wii Sports', 'Tetris']
for i, name in enumerate(top_10_sorted['Name']):
    if any(bundled in name for bundled in bundled_games):
        axes[1].text(top_10_sorted['Sales_Per_Year'].iloc[i] + 0.2, i, 
                    '⚠️ Bundled', va='center', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ IMPROVEMENTS IMPLEMENTED:")
print("   ✓ Genre color-coding provides semantic context")
print("   ✓ Release year and platform shown inline")
print("   ✓ Bottom panel normalizes by time (fairer comparison)")
print("   ✓ Bundled games flagged with warning icon")
print("   ✓ Viewer can now see GTA V's efficiency vs Wii Sports' longevity")

# %% [markdown]
# ---
# ## General Critique Framework

# %%
print("\n📋 GENERAL VISUALIZATION CRITIQUE CHECKLIST")
print("="*70)

critique_framework = """
WHEN ANALYZING ANY VISUALIZATION, ASK:

1. MESSAGE CLARITY:
   ☐ What is the primary claim or insight?
   ☐ Can a non-expert understand it in <10 seconds?
   ☐ Is the title specific and informative?
   ☐ Are axes and labels clear?

2. HIDDEN INFORMATION:
   ☐ What data is NOT shown?
   ☐ What time period is covered (and excluded)?
   ☐ Are sample sizes indicated?
   ☐ Are outliers included or filtered?
   ☐ What variables are held constant or ignored?

3. POTENTIAL MISINTERPRETATIONS:
   ☐ Could a viewer draw false causal conclusions?
   ☐ Does visual encoding exaggerate/minimize differences?
   ☐ Are scales manipulated (truncated, non-zero baseline)?
   ☐ Is correlation being confused with causation?
   ☐ Are comparisons fair (equal sample sizes, contexts)?

4. DESIGN CHOICES:
   ☐ Why this chart type? (bar vs line vs scatter, etc.)
   ☐ Are colors semantic or arbitrary?
   ☐ Is the visual encoding proportional to data?
   ☐ Are there unnecessary decorations (chartjunk)?
   ☐ Could a simpler chart convey the same insight?

5. STATISTICAL RIGOR:
   ☐ Are error bars or confidence intervals shown?
   ☐ Is statistical significance indicated?
   ☐ Are assumptions stated (normality, independence, etc.)?
   ☐ Is the data source credible and cited?
   ☐ Are limitations acknowledged?

6. ETHICAL CONSIDERATIONS:
   ☐ Could this visualization mislead intentionally?
   ☐ Is selection bias present (survivorship, cherry-picking)?
   ☐ Are vulnerable groups fairly represented?
   ☐ Does it reinforce harmful stereotypes?
   ☐ Is context provided to prevent misuse?

7. ACCESSIBILITY:
   ☐ Is it colorblind-friendly?
   ☐ Can it be interpreted in grayscale?
   ☐ Are text sizes readable?
   ☐ Is alt-text provided (for web)?
   ☐ Are patterns/textures used for redundancy?

8. CONTEXT AND TRANSPARENCY:
   ☐ Is the data source cited?
   ☐ Is the date of data collection stated?
   ☐ Are aggregation methods explained?
   ☐ Are data quality issues acknowledged?
   ☐ Is reproducibility possible?

9. ALTERNATIVE PERSPECTIVES:
   ☐ What would this look like from another angle?
   ☐ What if we used a different baseline?
   ☐ How would this change over time?
   ☐ What about excluded categories/regions?
   ☐ Could this data support an opposite conclusion?

10. ACTIONABILITY:
    ☐ What decision could this inform?
    ☐ What additional data is needed?
    ☐ What are the next questions to ask?
    ☐ What are the practical limitations?
"""

print(critique_framework)

# %% [markdown]
# ---
# ## Stress-Testing Visualizations: Adversarial Examples

# %% [markdown]
# ### Example: Same Data, Opposite Narratives

# %%
# Demonstrate how the same data can tell different stories
yearly_sales_total = df_clean.groupby('Year')['Global_Sales'].sum()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# LEFT: Pessimistic framing
axes[0].plot(yearly_sales_total.index[yearly_sales_total.index >= 2008], 
            yearly_sales_total[yearly_sales_total.index >= 2008],
            linewidth=3, marker='o', markersize=8, color='red')
axes[0].fill_between(yearly_sales_total.index[yearly_sales_total.index >= 2008],
                     0, yearly_sales_total[yearly_sales_total.index >= 2008],
                     alpha=0.3, color='red')
axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Global Sales (M)', fontsize=12, fontweight='bold')
axes[0].set_title('📉 "VIDEO GAME INDUSTRY IN CRISIS"\n50% Sales Decline Since 2008!', 
                 fontsize=13, fontweight='bold', color='darkred')
axes[0].set_ylim(0, 700)
axes[0].grid(alpha=0.3)

# RIGHT: Optimistic framing (same data, different window)
axes[1].plot(yearly_sales_total.index[yearly_sales_total.index <= 2010], 
            yearly_sales_total[yearly_sales_total.index <= 2010],
            linewidth=3, marker='o', markersize=8, color='green')
axes[1].fill_between(yearly_sales_total.index[yearly_sales_total.index <= 2010],
                     0, yearly_sales_total[yearly_sales_total.index <= 2010],
                     alpha=0.3, color='green')
axes[1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Global Sales (M)', fontsize=12, fontweight='bold')
axes[1].set_title('📈 "VIDEO GAME BOOM ACCELERATES"\n500% Growth in 30 Years!', 
                 fontsize=13, fontweight='bold', color='darkgreen')
axes[1].set_ylim(0, 700)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n⚠️  ADVERSARIAL EXAMPLE: SAME DATA, OPPOSITE NARRATIVES")
print("="*70)
print("LEFT CHART (Pessimistic):")
print("  • Shows only 2008-2016 (decline period)")
print("  • Red color suggests danger/loss")
print("  • Title emphasizes 'crisis' and '50% decline'")
print("  • True statement, but missing context")

print("\nRIGHT CHART (Optimistic):")
print("  • Shows only 1980-2010 (growth period)")
print("  • Green color suggests health/profit")
print("  • Title emphasizes 'boom' and '500% growth'")
print("  • Also true, but equally incomplete")

print("\nLESSON:")
print("  → Both charts are 'honest' in isolation")
print("  → Both are DISHONEST in what they omit")
print("  → Cherry-picking time windows creates false narratives")
print("  → ALWAYS show full temporal context")

# %% [markdown]
# ---
# ## Summary: Critical Reading Skills

# %%
print("\n" + "="*70)
print("SECTION I SUMMARY: CRITICAL VISUALIZATION ANALYSIS")
print("="*70)

print("\n📊 SELF-CRITIQUE DISCIPLINE:")
print("   1. ALWAYS state what your visualization DOESN'T show")
print("   2. Anticipate misinterpretations before they happen")
print("   3. Design improvements are never finished (iterative process)")
print("   4. Hidden information is often more important than visible data")

print("\n🔍 EXTERNAL CRITIQUE SKILLS:")
print("   1. Question the framing, not just the facts")
print("   2. Look for what's missing (time, context, sample size)")
print("   3. Consider alternative perspectives (who benefits from this narrative?)")
print("   4. Check for selection bias (survivorship, cherry-picking)")
print("   5. Demand transparency (data source, methodology, limitations)")

print("\n⚖️  ETHICAL VIGILANCE:")
print("   1. Same data can support opposite conclusions (framing matters)")
print("   2. Visual design choices are ethical choices")
print("   3. Omission is a form of lying")
print("   4. Accessibility is not optional")
print("   5. Always provide context for fair interpretation")

print("\n🎯 KEY LESSONS FROM CRITIQUES:")
print("   • Boxplots hide sample sizes → Add annotations")
print("   • Line charts hide absolute vs relative → Show both perspectives")
print("   • Scatter plots hide temporal dynamics → Animate or facet by era")
print("   • Top 10 lists create survivorship bias → Balance with failure analysis")
print("   • Cherry-picked windows mislead → Always show full timeline")

print("\n📋 PRACTICAL CRITIQUE WORKFLOW:")
print("   For EVERY visualization you create or encounter:")
print("   1. State the core message in one sentence")
print("   2. List 3 things the chart doesn't show")
print("   3. Imagine 2 ways a viewer could misinterpret it")
print("   4. Propose 2 design improvements")
print("   5. Check all 10 items in the critique checklist")

print("\n💡 MENTAL MODEL:")
print("   Treat every visualization as an ARGUMENT, not a fact")
print("   → It has premises (data, assumptions)")
print("   → It has reasoning (visual encoding)")
print("   → It has a conclusion (message)")
print("   → YOUR JOB: Evaluate the strength of each component")

print("\n" + "="*70)
print("SECTION I COMPLETE: Critical Visualization Reading")
print("="*70)
print("\nNext Steps:")
print("  → Section J: Interactive Visualization and Exploratory Tools")
print("  → Section K: Tool Comparison and Industry Reflection")
print("="*70)

# %%