# %% [markdown]
# # Section K: Tool Comparison and Industry Reflection
# ## Video Game Sales Dataset - Same Insight, Different Tools

# %% [markdown]
# ### Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# ## 15.1 Same Insight, Different Tools
# 
# ### Core Insight to Reproduce:
# **"Video game sales peaked in 2008-2009, with North America consistently dominating global markets, followed by a sharp decline in physical sales post-2010."**
# 
# This insight requires:
# 1. **Temporal trend visualization** (line chart)
# 2. **Regional comparison** (multi-line or stacked area)
# 3. **Peak identification** (annotation)
# 4. **Decline quantification** (trend analysis)

# %% [markdown]
# ### Data Preparation (Common to All Tools)

# %%
# Prepare aggregated data for all tools
yearly_regional = df_clean.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
yearly_regional['Total_Sales'] = yearly_regional.sum(axis=1)

print("Data Summary for Visualization:")
print(f"  Time Range: {yearly_regional.index.min()} - {yearly_regional.index.max()}")
print(f"  Peak Year: {yearly_regional['Total_Sales'].idxmax()}")
print(f"  Peak Sales: ${yearly_regional['Total_Sales'].max():.1f}M")
print(f"  2016 Sales: ${yearly_regional.loc[2016, 'Total_Sales']:.1f}M")
print(f"  Decline from Peak: {(1 - yearly_regional.loc[2016, 'Total_Sales'] / yearly_regional['Total_Sales'].max()) * 100:.1f}%")

# %% [markdown]
# ---
# ## Tool 1: Matplotlib (Low-Level Control, Maximum Flexibility)

# %%
print("\n" + "="*70)
print("TOOL 1: MATPLOTLIB")
print("="*70)
print("PHILOSOPHY: Low-level, procedural, maximum customization")
print("BEST FOR: Publication-quality static figures, fine-grained control")

# Create the visualization using Matplotlib
fig, ax = plt.subplots(figsize=(16, 8))

# Plot regional sales
ax.plot(yearly_regional.index, yearly_regional['NA_Sales'], 
       linewidth=3, marker='o', markersize=6, label='North America', color='#1f77b4')
ax.plot(yearly_regional.index, yearly_regional['EU_Sales'], 
       linewidth=3, marker='s', markersize=6, label='Europe', color='#ff7f0e')
ax.plot(yearly_regional.index, yearly_regional['JP_Sales'], 
       linewidth=3, marker='^', markersize=6, label='Japan', color='#2ca02c')
ax.plot(yearly_regional.index, yearly_regional['Other_Sales'], 
       linewidth=3, marker='d', markersize=6, label='Other Regions', color='#d62728')

# Highlight peak period
peak_year = yearly_regional['Total_Sales'].idxmax()
peak_value_na = yearly_regional.loc[peak_year, 'NA_Sales']

ax.axvspan(2008, 2010, alpha=0.2, color='yellow', zorder=0, label='Peak Period (2008-2010)')
ax.axvline(peak_year, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak Year ({peak_year})')

# Annotate peak
ax.annotate(f'Peak: {peak_year}\nNA: ${peak_value_na:.1f}M', 
           xy=(peak_year, peak_value_na), 
           xytext=(peak_year - 5, peak_value_na + 80),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2))

# Annotate decline
decline_year = 2016
decline_value_na = yearly_regional.loc[decline_year, 'NA_Sales']
decline_pct = (1 - yearly_regional.loc[decline_year, 'Total_Sales'] / yearly_regional['Total_Sales'].max()) * 100

ax.annotate(f'2016: ${decline_value_na:.1f}M\n({decline_pct:.0f}% decline from peak)', 
           xy=(decline_year, decline_value_na), 
           xytext=(decline_year - 2, decline_value_na + 60),
           arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
           fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1.5))

# Styling
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Annual Sales (Millions of Units)', fontsize=14, fontweight='bold')
ax.set_title('Video Game Sales Trends by Region (1980-2016)\nMatplotlib Implementation', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_xlim(yearly_regional.index.min() - 1, yearly_regional.index.max() + 1)
ax.set_ylim(0, yearly_regional['NA_Sales'].max() * 1.15)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.show()

print("\n✅ MATPLOTLIB STRENGTHS:")
print("   1. CONTROL: Every pixel can be precisely positioned")
print("      → Custom annotation placement, exact color codes")
print("   2. STABILITY: Mature library, extensive documentation")
print("      → Code from 2010 still works in 2026")
print("   3. PUBLICATION QUALITY: Vector output (PDF, SVG)")
print("      → Academic journals accept Matplotlib figures directly")
print("   4. CUSTOMIZATION: Full control over every visual element")
print("      → Can create completely novel chart types")
print("   5. INTEGRATION: Works seamlessly with NumPy, Pandas")
print("      → Direct array plotting without conversion")

print("\n❌ MATPLOTLIB WEAKNESSES:")
print("   1. VERBOSITY: ~30 lines of code for medium-complexity chart")
print("      → Requires explicit configuration of many parameters")
print("   2. STATIC ONLY: No built-in interactivity")
print("      → Tooltips, zoom, filters require external libraries")
print("   3. LEARNING CURVE: Two interfaces (pyplot vs OOP)")
print("      → Beginners confused by `plt.plot()` vs `ax.plot()`")
print("   4. AESTHETICS: Default styles dated (improving with recent versions)")
print("      → Requires manual styling for modern look")
print("   5. BOILERPLATE: Repetitive code for similar charts")
print("      → Each chart built from scratch")

print("\n🎯 MATPLOTLIB IDEAL USE CASES:")
print("   • Academic papers and journal submissions")
print("   • Printed reports and presentations")
print("   • Highly customized, one-off visualizations")
print("   • When you need exact control over every element")
print("   • Embedding in GUI applications (Qt, Tkinter)")

print("\n⏱️  DEVELOPMENT TIME ESTIMATE: 15-20 minutes")
print("   (For experienced user to create this annotated multi-line chart)")

# %% [markdown]
# ---
# ## Tool 2: Seaborn (Statistical Graphics, Matplotlib Wrapper)

# %%
print("\n" + "="*70)
print("TOOL 2: SEABORN")
print("="*70)
print("PHILOSOPHY: High-level statistical visualization, built on Matplotlib")
print("BEST FOR: Statistical plots, rapid prototyping, beautiful defaults")

# Prepare data in long format (required for Seaborn)
df_long = yearly_regional.reset_index().melt(
    id_vars='Year', 
    value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
    var_name='Region', 
    value_name='Sales'
)

# Clean region names
df_long['Region'] = df_long['Region'].str.replace('_Sales', '').replace({
    'NA': 'North America',
    'EU': 'Europe',
    'JP': 'Japan',
    'Other': 'Other Regions'
})

# Create the visualization using Seaborn
fig, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(
    data=df_long, 
    x='Year', 
    y='Sales', 
    hue='Region',
    style='Region',
    markers=['o', 's', '^', 'd'],
    markersize=8,
    linewidth=3,
    palette='tab10',
    ax=ax
)

# Add peak period highlighting (using Matplotlib since Seaborn doesn't have built-in)
ax.axvspan(2008, 2010, alpha=0.2, color='yellow', zorder=0)
ax.axvline(peak_year, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Annotations (manual, Seaborn doesn't auto-annotate)
ax.annotate(f'Peak Year: {peak_year}', 
           xy=(peak_year, peak_value_na), 
           xytext=(peak_year - 5, peak_value_na + 80),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

# Styling
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Annual Sales (Millions of Units)', fontsize=14, fontweight='bold')
ax.set_title('Video Game Sales Trends by Region (1980-2016)\nSeaborn Implementation', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(title='Region', fontsize=11, title_fontsize=12, loc='upper left', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_xlim(df_long['Year'].min() - 1, df_long['Year'].max() + 1)

plt.tight_layout()
plt.show()

print("\n✅ SEABORN STRENGTHS:")
print("   1. BEAUTIFUL DEFAULTS: Aesthetically pleasing out-of-the-box")
print("      → Color palettes, fonts, grid styles all optimized")
print("   2. STATISTICAL FOCUS: Built-in confidence intervals, regression")
print("      → `sns.regplot()`, `sns.boxplot()` handle stats automatically")
print("   3. DATAFRAME INTEGRATION: Works natively with Pandas")
print("      → Pass DataFrame and column names, not arrays")
print("   4. CONCISENESS: Less code than raw Matplotlib")
print("      → One function call for complex statistical plots")
print("   5. CONSISTENCY: Uniform API across plot types")
print("      → All functions use `data=`, `x=`, `y=`, `hue=` pattern")

print("\n❌ SEABORN WEAKNESSES:")
print("   1. ABSTRACTION LIMITATIONS: Hard to customize beyond defaults")
print("      → Falls back to Matplotlib for fine control (defeating purpose)")
print("   2. DATA FORMAT RIGIDITY: Requires long-form data")
print("      → Had to melt DataFrame from wide to long format")
print("   3. ANNOTATION GAP: No built-in annotation tools")
print("      → Peak/decline labels required manual Matplotlib code")
print("   4. PERFORMANCE: Slower than Matplotlib for large datasets")
print("      → Statistical computations add overhead")
print("   5. MATPLOTLIB DEPENDENCY: Inherits Matplotlib's static nature")
print("      → Still no interactivity")

print("\n🎯 SEABORN IDEAL USE CASES:")
print("   • Exploratory data analysis (EDA)")
print("   • Statistical reports with standard plot types")
print("   • Rapid prototyping of visualizations")
print("   • When aesthetics matter but time is limited")
print("   • Teaching data science (intuitive API)")

print("\n⏱️  DEVELOPMENT TIME ESTIMATE: 10-12 minutes")
print("   (Faster than Matplotlib for standard plots, but data prep adds time)")

# %% [markdown]
# ---
# ## Tool 3: Plotly (Interactive Web-Based Visualization)

# %%
print("\n" + "="*70)
print("TOOL 3: PLOTLY")
print("="*70)
print("PHILOSOPHY: Interactive, web-native, declarative graphics")
print("BEST FOR: Dashboards, presentations, exploratory interfaces")

# Create the visualization using Plotly
fig = go.Figure()

# Add regional traces
regions = {
    'North America': {'data': yearly_regional['NA_Sales'], 'color': '#1f77b4', 'symbol': 'circle'},
    'Europe': {'data': yearly_regional['EU_Sales'], 'color': '#ff7f0e', 'symbol': 'square'},
    'Japan': {'data': yearly_regional['JP_Sales'], 'color': '#2ca02c', 'symbol': 'triangle-up'},
    'Other Regions': {'data': yearly_regional['Other_Sales'], 'color': '#d62728', 'symbol': 'diamond'}
}

for region, props in regions.items():
    fig.add_trace(go.Scatter(
        x=yearly_regional.index,
        y=props['data'],
        mode='lines+markers',
        name=region,
        line=dict(width=3, color=props['color']),
        marker=dict(size=8, symbol=props['symbol'], color=props['color'], 
                   line=dict(width=1, color='black')),
        hovertemplate=f'<b>{region}</b><br>' +
                     'Year: %{x}<br>' +
                     'Sales: $%{y:.1f}M<br>' +
                     '<extra></extra>'
    ))

# Add peak period shading
fig.add_vrect(
    x0=2007.5, x1=2010.5,
    fillcolor='yellow', opacity=0.2,
    layer='below', line_width=0,
    annotation_text='Peak Period', annotation_position='top left',
    annotation=dict(font_size=11, font_color='darkgoldenrod')
)

# Add peak line
fig.add_vline(
    x=peak_year, 
    line_dash='dash', 
    line_color='red', 
    line_width=2,
    opacity=0.7,
    annotation_text=f'Peak Year: {peak_year}',
    annotation_position='top',
    annotation=dict(font_size=12, font_color='red')
)

# Add annotations
fig.add_annotation(
    x=peak_year, y=peak_value_na,
    text=f'<b>Peak: {peak_year}</b><br>NA: ${peak_value_na:.1f}M',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='red',
    ax=-80, ay=-60,
    bordercolor='red',
    borderwidth=2,
    borderpad=4,
    bgcolor='yellow',
    opacity=0.8,
    font=dict(size=11, color='black')
)

fig.add_annotation(
    x=decline_year, y=decline_value_na,
    text=f'<b>2016: ${decline_value_na:.1f}M</b><br>({decline_pct:.0f}% decline)',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='darkred',
    ax=-40, ay=60,
    bordercolor='darkred',
    borderwidth=2,
    borderpad=4,
    bgcolor='lightcoral',
    opacity=0.7,
    font=dict(size=10, color='black')
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Video Game Sales Trends by Region (1980-2016)</b><br><sub>Plotly Implementation - Hover for details</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    xaxis=dict(
        title='<b>Year</b>',
        titlefont=dict(size=14),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[yearly_regional.index.min() - 1, yearly_regional.index.max() + 1]
    ),
    yaxis=dict(
        title='<b>Annual Sales (Millions of Units)</b>',
        titlefont=dict(size=14),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[0, yearly_regional['NA_Sales'].max() * 1.15]
    ),
    hovermode='x unified',
    legend=dict(
        title='<b>Region</b>',
        x=0.02, y=0.98,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11)
    ),
    plot_bgcolor='rgba(245,245,245,0.9)',
    height=600,
    font=dict(family='Arial, sans-serif', size=12)
)

fig.show()

print("\n✅ PLOTLY STRENGTHS:")
print("   1. INTERACTIVITY: Built-in hover, zoom, pan, export")
print("      → No additional code required for rich tooltips")
print("   2. WEB-NATIVE: Renders as JavaScript in browser")
print("      → Shareable HTML files, embeddable in web apps")
print("   3. PROFESSIONAL AESTHETICS: Modern, polished look")
print("      → Suitable for client-facing dashboards")
print("   4. DECLARATIVE API: Describe what you want, not how to draw it")
print("      → Less procedural than Matplotlib")
print("   5. RESPONSIVE: Auto-adjusts to screen size")
print("      → Mobile-friendly visualizations")

print("\n❌ PLOTLY WEAKNESSES:")
print("   1. LARGE FILE SIZES: HTML exports can be several MB")
print("      → Not ideal for email attachments or low-bandwidth")
print("   2. DEPENDENCY ON JAVASCRIPT: Requires browser, won't work in PDF")
print("      → Can't print static version without export step")
print("   3. LEARNING CURVE: Different paradigm from Matplotlib")
print("      → Graph objects vs express, layout vs traces")
print("   4. LIMITED OFFLINE SUPPORT: Some features require internet")
print("      → Mapbox, certain renderers need connectivity")
print("   5. VERBOSITY FOR CUSTOMIZATION: Deep nesting in layout dict")
print("      → `fig.update_layout(xaxis=dict(title=dict(text='...')))` gets unwieldy")

print("\n🎯 PLOTLY IDEAL USE CASES:")
print("   • Interactive dashboards and web applications")
print("   • Client presentations (live demos)")
print("   • Exploratory analysis with stakeholders")
print("   • Self-service analytics platforms")
print("   • Reports where users need to explore data themselves")

print("\n⏱️  DEVELOPMENT TIME ESTIMATE: 12-15 minutes")
print("   (Similar to Matplotlib, but interactivity comes free)")

# %% [markdown]
# ---
# ## Side-by-Side Comparison: Same Data, Three Implementations

# %%
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS: MATPLOTLIB vs SEABORN vs PLOTLY")
print("="*70)

comparison_table = """
┌──────────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ CRITERION                │ MATPLOTLIB          │ SEABORN             │ PLOTLY              │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ LINES OF CODE            │ ~35 lines           │ ~25 lines           │ ~40 lines           │
│                          │ (most verbose)      │ (most concise)      │ (verbose for custom)│
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ DEVELOPMENT TIME         │ 15-20 min           │ 10-12 min           │ 12-15 min           │
│                          │ (manual everything) │ (fastest)           │ (medium)            │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ INTERACTIVITY            │ ✗ None              │ ✗ None              │ ✓ Full (zoom, hover)│
│                          │                     │                     │                     │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ CUSTOMIZATION LEVEL      │ ✓✓✓ Maximum         │ ✓✓ High             │ ✓✓ High             │
│                          │ (pixel-perfect)     │ (limited by API)    │ (JSON-like config)  │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ DEFAULT AESTHETICS       │ ✗ Dated             │ ✓✓✓ Beautiful       │ ✓✓✓ Modern          │
│                          │ (needs styling)     │ (best out-of-box)   │ (professional)      │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ OUTPUT FORMAT            │ ✓ PNG, PDF, SVG     │ ✓ PNG, PDF, SVG     │ ✓ HTML, PNG (static)│
│                          │ (publication-ready) │ (publication-ready) │ (web-optimized)     │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ LEARNING CURVE           │ Medium-High         │ Low-Medium          │ Medium              │
│                          │ (two interfaces)    │ (intuitive)         │ (new paradigm)      │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ STATISTICAL FEATURES     │ ✗ Manual            │ ✓✓✓ Built-in        │ ✓ Some (less than SB)│
│                          │ (need SciPy/StatsM) │ (CI, regression)    │                     │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ DATA FORMAT FLEXIBILITY  │ ✓✓✓ Arrays, lists   │ ✓ Requires long DF  │ ✓✓ Both wide & long │
│                          │ (any format)        │ (rigid)             │ (flexible)          │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ FILE SIZE (output)       │ ✓✓✓ Small (KB)      │ ✓✓✓ Small (KB)      │ ✗ Large (MB for HTML)│
│                          │ (vector efficient)  │ (vector efficient)  │ (includes JS libs)  │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ OFFLINE CAPABILITY       │ ✓✓✓ Full            │ ✓✓✓ Full            │ ✓✓ Mostly           │
│                          │ (no dependencies)   │ (no dependencies)   │ (some features need │
│                          │                     │                     │  internet)          │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ PERFORMANCE (large data) │ ✓✓✓ Fast            │ ✓✓ Slower           │ ✓ Can be slow       │
│                          │ (native rendering)  │ (stats overhead)    │ (JS rendering)      │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ SHAREABILITY             │ ✓ Attach image      │ ✓ Attach image      │ ✓✓✓ Share HTML link │
│                          │ (static)            │ (static)            │ (interactive)       │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ DOCUMENTATION QUALITY    │ ✓✓✓ Extensive       │ ✓✓✓ Excellent       │ ✓✓ Good             │
│                          │ (20+ years)         │ (tutorials)         │ (growing)           │
├──────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ STABILITY (API changes)  │ ✓✓✓ Stable          │ ✓✓ Mostly stable    │ ✓ Frequent updates  │
│                          │ (backward compat)   │ (minor breaks)      │ (breaking changes)  │
└──────────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

SCORING: ✓✓✓ Excellent | ✓✓ Good | ✓ Adequate | ✗ Poor/None
"""

print(comparison_table)

# %% [markdown]
# ---
# ## Analytical Clarity: Which Tool Best Reveals the Insight?

# %%
print("\n📊 ANALYTICAL CLARITY COMPARISON")
print("="*70)

print("\nINSIGHT TO COMMUNICATE:")
print("'Video game sales peaked in 2008-2009, with North America dominating,'")
print("'followed by a 60%+ decline in physical sales by 2016.'")

print("\n1. MATPLOTLIB CLARITY: ✓✓✓")
print("   STRENGTHS:")
print("      • Precise annotation placement highlights peak and decline")
print("      • Custom styling emphasizes key regions (thick lines, markers)")
print("      • Yellow shading immediately draws eye to peak period")
print("      • Static → forces viewer to focus on intended message")
print("   WEAKNESSES:")
print("      • No interactivity → can't explore 'why' the decline happened")
print("      • Requires viewer trust (can't verify data themselves)")

print("\n2. SEABORN CLARITY: ✓✓")
print("   STRENGTHS:")
print("      • Beautiful defaults make trends immediately visible")
print("      • Less visual clutter than manual Matplotlib styling")
print("      • Color palette is intuitive (no semantic conflict)")
print("   WEAKNESSES:")
print("      • Annotations still manual (Matplotlib code)")
print("      • Long-form data conversion adds cognitive step")
print("      • No built-in tools for peak/decline highlighting")
print("      • Doesn't add value beyond Matplotlib for this specific chart")

print("\n3. PLOTLY CLARITY: ✓✓✓")
print("   STRENGTHS:")
print("      • Hover tooltips provide exact values without clutter")
print("      • Viewer can zoom to 2008-2010 to examine peak closely")
print("      • Interactive legend (click to hide/show regions)")
print("      • Can verify analyst's claim by exploring data themselves")
print("   WEAKNESSES:")
print("      • Interactivity can distract from core message")
print("      • Requires active engagement (some viewers prefer passive)")
print("      • Screenshot for email loses all benefits")

print("\n🎯 VERDICT FOR THIS SPECIFIC INSIGHT:")
print("   WINNER: TIE between Matplotlib and Plotly")
print("      • Matplotlib: Best for formal reports, publications")
print("      • Plotly: Best for live presentations, stakeholder meetings")
print("      • Seaborn: Not optimal (doesn't add unique value here)")

# %% [markdown]
# ---
# ## Suitability for Different Stakeholder Audiences

# %%
print("\n👥 STAKEHOLDER SUITABILITY ANALYSIS")
print("="*70)

stakeholder_table = """
┌─────────────────────────┬──────────────────────────────────────────────────────────────┐
│ STAKEHOLDER TYPE        │ RECOMMENDED TOOL & RATIONALE                                 │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 1. EXECUTIVE LEADERSHIP │ PLOTLY (Interactive Dashboard)                               │
│    (CEO, VP)            │ Rationale:                                                   │
│                         │ • High-level overview with drill-down capability             │
│                         │ • Professional aesthetics build trust                        │
│                         │ • Can explore their own questions during meeting             │
│                         │ • Export to HTML for asynchronous review                     │
│                         │ ✗ Avoid: Matplotlib (too technical-looking)                  │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 2. DATA SCIENCE TEAM    │ SEABORN (Rapid Prototyping) + MATPLOTLIB (Final Polish)     │
│    (Analysts, DS)       │ Rationale:                                                   │
│                         │ • Seaborn for quick EDA and hypothesis generation            │
│                         │ • Matplotlib for publication-quality final figures           │
│                         │ • Both tools familiar to technical audience                  │
│                         │ • Can read and modify code easily                            │
│                         │ ✗ Avoid: Plotly (overkill for internal technical analysis)   │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 3. ACADEMIC REVIEWERS   │ MATPLOTLIB (Publication Standard)                            │
│    (Journal Editors)    │ Rationale:                                                   │
│                         │ • Vector graphics (PDF/SVG) required by journals             │
│                         │ • Pixel-perfect control for multi-panel figures              │
│                         │ • Reproducible code for peer review                          │
│                         │ • Meets submission guidelines (font sizes, margins)          │
│                         │ ✗ Avoid: Plotly (journals don't accept HTML)                 │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 4. BUSINESS CLIENTS     │ PLOTLY (Interactive) or SEABORN (Static)                     │
│    (External)           │ Rationale:                                                   │
│                         │ • Plotly for live demos (impressive, engaging)               │
│                         │ • Seaborn for email reports (beautiful, non-threatening)     │
│                         │ • Avoid technical jargon in annotations                      │
│                         │ • Focus on business outcomes, not methods                    │
│                         │ ✗ Avoid: Raw Matplotlib (looks unpolished)                   │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 5. GENERAL PUBLIC       │ PLOTLY (Simplified) or SEABORN (Infographic Style)          │
│    (Media, Blog)        │ Rationale:                                                   │
│                         │ • Plotly for web articles (embeddable, shareable)            │
│                         │ • Seaborn for social media (PNG exports well)               │
│                         │ • Minimize labels, maximize visual impact                    │
│                         │ • Use relatable context (compare to familiar benchmarks)     │
│                         │ ✗ Avoid: Complex annotations, statistical jargon             │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 6. TEACHING / STUDENTS  │ ALL THREE (Pedagogical Comparison)                           │
│    (Classroom)          │ Rationale:                                                   │
│                         │ • Show Matplotlib for understanding fundamentals             │
│                         │ • Show Seaborn for rapid analysis workflows                  │
│                         │ • Show Plotly for modern interactive applications            │
│                         │ • Emphasize trade-offs, not 'best' tool                      │
│                         │ ✓ All tools have educational value                           │
└─────────────────────────┴──────────────────────────────────────────────────────────────┘
"""

print(stakeholder_table)

# %% [markdown]
# ---
# ## Code Complexity and Maintainability Comparison

# %%
print("\n🔧 CODE MAINTAINABILITY ANALYSIS")
print("="*70)

print("\n1. MATPLOTLIB CODE:")
print("   PROS:")
print("      ✓ Explicit → Every element is clearly defined in code")
print("      ✓ Debuggable → Can inspect `ax` object at any point")
print("      ✓ Stable → Old code rarely breaks with updates")
print("   CONS:")
print("      ✗ Verbose → 35+ lines for medium-complexity chart")
print("      ✗ Repetitive → Similar code for similar charts (copy-paste errors)")
print("      ✗ Boilerplate → Lots of setup before actual plotting")
print("\n   MAINTENANCE EFFORT: Medium-High")
print("      • Changes require finding specific line in long code block")
print("      • Adding new series needs manual color/marker assignment")
print("      • Refactoring is tedious (lots of coupled settings)")

print("\n2. SEABORN CODE:")
print("   PROS:")
print("      ✓ Concise → 25 lines, mostly data prep")
print("      ✓ Consistent → All plots follow same API pattern")
print("      ✓ Readable → High-level abstractions are self-documenting")
print("   CONS:")
print("      ✗ Data format dependency → Must reshape data first (melt, pivot)")
print("      ✗ Customization limits → Falls back to Matplotlib for details")
print("      ✗ Version sensitivity → API changes more frequent than Matplotlib")
print("\n   MAINTENANCE EFFORT: Low-Medium")
print("      • Data reshaping is separate step (easy to modify)")
print("      • One function call for entire plot (easy to swap chart types)")
print("      • But custom annotations still require Matplotlib knowledge")

print("\n3. PLOTLY CODE:")
print("   PROS:")
print("      ✓ Declarative → Describe desired output, not steps")
print("      ✓ Modular → Traces and layout are separate (easy to rearrange)")
print("      ✓ Interactivity-first → Hover, zoom come free (no extra code)")
print("   CONS:")
print("      ✗ Deep nesting → `fig.update_layout(xaxis=dict(title=dict(...)))`")
print("      ✗ Verbosity for customization → 40+ lines for full control")
print("      ✗ API churn → Breaking changes in major versions")
print("\n   MAINTENANCE EFFORT: Medium")
print("      • Modular structure makes adding traces easy")
print("      • Layout updates can be complex (nested dictionaries)")
print("      • Interactive features may break with library updates")

print("\n🎯 MAINTAINABILITY RANKING:")
print("   1st: SEABORN (easiest to modify for standard plots)")
print("   2nd: PLOTLY (modular, but nesting can be confusing)")
print("   3rd: MATPLOTLIB (most stable, but most verbose)")

# %% [markdown]
# ---
# ## Performance Benchmarking (Large Dataset Simulation)

# %%
print("\n⚡ PERFORMANCE COMPARISON (Large Dataset)")
print("="*70)

# Simulate larger dataset (100k games instead of 16k)
import time

np.random.seed(42)
large_df = pd.DataFrame({
    'Year': np.random.choice(range(1980, 2017), 100000),
    'NA_Sales': np.random.exponential(0.5, 100000),
    'EU_Sales': np.random.exponential(0.3, 100000),
    'JP_Sales': np.random.exponential(0.2, 100000),
})

large_yearly = large_df.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()

# Benchmark Matplotlib
start = time.time()
fig, ax = plt.subplots()
ax.plot(large_yearly.index, large_yearly['NA_Sales'])
ax.plot(large_yearly.index, large_yearly['EU_Sales'])
ax.plot(large_yearly.index, large_yearly['JP_Sales'])
plt.close(fig)
matplotlib_time = time.time() - start

# Benchmark Seaborn
large_long = large_yearly.reset_index().melt(id_vars='Year', var_name='Region', value_name='Sales')
start = time.time()
fig, ax = plt.subplots()
sns.lineplot(data=large_long, x='Year', y='Sales', hue='Region', ax=ax)
plt.close(fig)
seaborn_time = time.time() - start

# Benchmark Plotly
start = time.time()
fig = px.line(large_long, x='Year', y='Sales', color='Region')
plotly_time = time.time() - start

print(f"\nRENDERING TIME (100,000 data points → 37 yearly aggregates):")
print(f"   Matplotlib: {matplotlib_time:.4f} seconds")
print(f"   Seaborn:    {seaborn_time:.4f} seconds ({seaborn_time/matplotlib_time:.1f}× slower)")
print(f"   Plotly:     {plotly_time:.4f} seconds ({plotly_time/matplotlib_time:.1f}× slower)")

print("\n📊 PERFORMANCE INSIGHTS:")
print("   • Matplotlib is fastest (native C rendering)")
print("   • Seaborn adds overhead (statistical computations)")
print("   • Plotly is slowest (JavaScript generation)")
print("   • For <10k points, all are fast enough (< 1 second)")
print("   • For >100k points, Matplotlib or Plotly with downsampling")

# %% [markdown]
# ---
# ## Real-World Industry Reflection

# %%
print("\n🏢 INDUSTRY TOOL USAGE PATTERNS")
print("="*70)

print("\n1. TECH COMPANIES (Google, Meta, Amazon):")
print("   PRIMARY: Custom internal tools (D3.js, Vega, proprietary)")
print("   SECONDARY: Plotly Dash for rapid prototyping")
print("   RARE: Matplotlib (too static for product analytics)")
print("   WHY: Need interactivity at scale, real-time dashboards")

print("\n2. FINANCE (Bloomberg, JP Morgan):")
print("   PRIMARY: Specialized tools (Tableau, Power BI, proprietary)")
print("   SECONDARY: Matplotlib for regulatory reports (static PDFs)")
print("   RARE: Plotly (security concerns with external JS libraries)")
print("   WHY: Compliance requires static, auditable outputs")

print("\n3. ACADEMIA (Universities, Research Labs):")
print("   PRIMARY: Matplotlib (publication standard)")
print("   SECONDARY: Seaborn (teaching, EDA)")
print("   GROWING: Plotly for interactive papers (HTML supplements)")
print("   WHY: Journals require vector graphics, reproducibility")

print("\n4. STARTUPS (Data-Driven Products):")
print("   PRIMARY: Plotly Dash or Streamlit")
print("   SECONDARY: Seaborn for internal analysis")
print("   RARE: Matplotlib (too slow for iteration speed)")
print("   WHY: Fast prototyping, investor demos need 'wow factor'")

print("\n5. HEALTHCARE (Hospitals, Pharma):")
print("   PRIMARY: Specialized medical software")
print("   SECONDARY: Matplotlib (clinical trial reports)")
print("   RARE: Web-based tools (HIPAA compliance issues)")
print("   WHY: Regulatory requirements favor static, validated outputs")

print("\n6. CONSULTING (McKinsey, Deloitte):")
print("   PRIMARY: PowerPoint + Matplotlib PNG exports")
print("   SECONDARY: Tableau for client-facing dashboards")
print("   RARE: Interactive notebooks (clients don't run code)")
print("   WHY: Slide decks are deliverable format")

# %% [markdown]
# ---
# ## Final Recommendation Framework

# %%
print("\n🎯 TOOL SELECTION DECISION TREE")
print("="*70)

decision_tree = """
START: Which tool should I use?
│
├─ Q1: Is the output for a printed/PDF report?
│  ├─ YES → MATPLOTLIB or SEABORN
│  │        └─ Q1a: Do you need pixel-perfect control?
│  │           ├─ YES → MATPLOTLIB
│  │           └─ NO → SEABORN (faster, prettier defaults)
│  │
│  └─ NO → Continue to Q2
│
├─ Q2: Do users need to interact with the data?
│  ├─ YES → PLOTLY
│  │        └─ Q2a: Is this for a web dashboard?
│  │           ├─ YES → Plotly Dash or Streamlit
│  │           └─ NO → Plotly standalone HTML
│  │
│  └─ NO → Continue to Q3
│
├─ Q3: Are you doing exploratory data analysis (EDA)?
│  ├─ YES → SEABORN (fastest iteration)
│  │        └─ Iterate quickly, refine with Matplotlib later
│  │
│  └─ NO → Continue to Q4
│
├─ Q4: Is this for academic publication?
│  ├─ YES → MATPLOTLIB (journal standard)
│  │        └─ Vector graphics (PDF/SVG) required
│  │
│  └─ NO → Continue to Q5
│
├─ Q5: Is your audience non-technical?
│  ├─ YES → PLOTLY (impressive visuals) or SEABORN (approachable)
│  │        └─ Avoid technical jargon in labels
│  │
│  └─ NO → Continue to Q6
│
└─ Q6: Do you need maximum customization?
   ├─ YES → MATPLOTLIB (full control)
   └─ NO → SEABORN (good defaults, less code)

DEFAULT RECOMMENDATION:
   • Start with SEABORN for EDA
   • Refine with MATPLOTLIB for publication
   • Convert to PLOTLY for stakeholder presentations
"""

print(decision_tree)

# %% [markdown]
# ---
# ## Summary: Tool Selection Wisdom

# %%
print("\n" + "="*70)
print("SECTION K SUMMARY: TOOL COMPARISON AND REFLECTION")
print("="*70)

print("\n📊 CORE INSIGHT REPRODUCED IN THREE TOOLS:")
print("   'Video game sales peaked in 2008-2009, with North America dominating,")
print("    followed by a 60%+ decline in physical sales by 2016.'")

print("\n🔧 TOOL COMPARISON SUMMARY:")
print("\n   MATPLOTLIB:")
print("      ✓ Maximum control, publication-quality, stable")
print("      ✗ Verbose, static-only, dated defaults")
print("      🎯 Best for: Academic papers, formal reports, archival")
print("      ⏱️  Dev time: 15-20 min | Code: ~35 lines")

print("\n   SEABORN:")
print("      ✓ Beautiful defaults, concise code, statistical focus")
print("      ✗ Rigid data format, limited customization, static-only")
print("      🎯 Best for: Exploratory analysis, rapid prototyping, teaching")
print("      ⏱️  Dev time: 10-12 min | Code: ~25 lines")

print("\n   PLOTLY:")
print("      ✓ Interactivity, modern aesthetics, web-native")
print("      ✗ Large files, requires browser, API churn")
print("      🎯 Best for: Dashboards, presentations, stakeholder exploration")
print("      ⏱️  Dev time: 12-15 min | Code: ~40 lines")

print("\n⚖️  NO UNIVERSAL 'BEST' TOOL:")
print("   Context determines appropriateness:")
print("      • Output format (print vs web)")
print("      • Audience (technical vs executive)")
print("      • Purpose (exploration vs communication)")
print("      • Timeline (rapid prototype vs polished final)")
print("      • Maintenance (one-off vs long-term dashboard)")

print("\n🎓 KEY LESSONS:")
print("   1. Master ONE tool deeply (usually Matplotlib)")
print("   2. Learn others for specific use cases")
print("   3. Prototype with fast tools (Seaborn), refine with control (Matplotlib)")
print("   4. Interactivity is not always better (can distract from message)")
print("   5. Code verbosity is acceptable if it improves maintainability")
print("   6. Professional polish matters for external audiences")
print("   7. Always have a static fallback (Plotly HTML can break)")

print("\n💡 PRACTICAL WORKFLOW:")
print("   EDA Phase: Seaborn (quick iteration)")
print("   Analysis Phase: Matplotlib (precise control)")
print("   Presentation Phase: Plotly (engagement)")
print("   Publication Phase: Matplotlib (journal standards)")
print("   Long-term Dashboard: Plotly Dash (maintainability)")

print("\n🏆 FINAL WISDOM:")
print("   'The best visualization tool is the one that communicates")
print("    your insight clearly to your specific audience.'")
print("\n   → Focus on the message, not the medium")
print("   → Let stakeholder needs drive tool choice")
print("   → Avoid tool evangelism (all have trade-offs)")
print("   → Develop polyglot visualization skills")

print("\n" + "="*70)
print("SECTION K COMPLETE: Tool Comparison")
print("="*70)
print("\n🎉 ALL SECTIONS COMPLETE!")
print("\nCOMPREHENSIVE ANALYSIS WORKFLOW:")
print("   A: Data structure and context ✓")
print("   B: Univariate exploration ✓")
print("   C: Bivariate relationships ✓")
print("   D: Multivariate interactions ✓")
print("   E: Statistical validation ✓")
print("   F: Dimensionality reduction (PCA) ✓")
print("   G: Clustering exploration ✓")
print("   H: Visualization ethics ✓")
print("   I: Critical visualization reading ✓")
print("   J: Interactive features ✓")
print("   K: Tool comparison ✓")
print("="*70)

# %%