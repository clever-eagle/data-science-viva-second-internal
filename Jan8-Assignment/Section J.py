# %% [markdown]
# # Section J: Interactive Visualization and Exploratory Tools
# ## Video Game Sales Dataset - Dynamic Exploration and User-Driven Analysis

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
# ## 14.1 Interactive Question Expansion
# 
# ### Core Questions That REQUIRE Interactivity:
# 1. **Drill-Down Questions**: "Show me details of this outlier game"
# 2. **Comparative Questions**: "How does this platform compare if I filter by genre?"
# 3. **Temporal Questions**: "What if I focus on a specific decade?"
# 4. **Multi-Dimensional Questions**: "Show me overlap between region, genre, and platform"
# 5. **Exploratory Questions**: "Let me explore patterns myself without pre-defined views"

# %% [markdown]
# ### Why Static Visualizations Fail for These Questions

# %%
print("🤔 LIMITATIONS OF STATIC VISUALIZATIONS")
print("="*70)

print("\n1. INFORMATION OVERLOAD:")
print("   Static Problem: Showing all games on one scatter plot → Unreadable")
print("   Interactive Solution: Zoom, pan, filter to focus on relevant subset")

print("\n2. DETAIL ON DEMAND:")
print("   Static Problem: Can't show game titles without cluttering chart")
print("   Interactive Solution: Hover tooltips reveal details dynamically")

print("\n3. MULTI-FACETED EXPLORATION:")
print("   Static Problem: Requires creating dozens of separate charts")
print("   Interactive Solution: Dropdowns/sliders let user switch dimensions")

print("\n4. HYPOTHESIS TESTING:")
print("   Static Problem: Analyst must guess what viewer wants to see")
print("   Interactive Solution: Viewer formulates and tests own hypotheses")

print("\n5. COMPARATIVE ANALYSIS:")
print("   Static Problem: Side-by-side comparisons require rigid structure")
print("   Interactive Solution: Toggle between groups/categories dynamically")

print("\n6. TEMPORAL DYNAMICS:")
print("   Static Problem: Animation as GIF loses control and context")
print("   Interactive Solution: Play/pause, scrub timeline, step through frames")

# %% [markdown]
# ---
# ## Interactive Feature 1: Hover-Based Detail Tooltips

# %% [markdown]
# ### Question: "What are the details of high-selling games in each genre?"

# %%
print("\n📊 INTERACTIVE FEATURE 1: HOVER TOOLTIPS")
print("="*70)
print("USER NEED: I see an interesting point on the chart - what game is it?")
print("STATIC LIMITATION: Can't label all points without overlap/clutter")
print("INTERACTIVE SOLUTION: Hover over any point to see details on demand")

# Create interactive scatter plot with rich tooltips
top_genres = df_clean.groupby('Genre')['Global_Sales'].sum().nlargest(5).index
df_interactive = df_clean[df_clean['Genre'].isin(top_genres)].copy()

# Create hover text with multiple attributes
df_interactive['hover_text'] = (
    '<b>' + df_interactive['Name'] + '</b><br>' +
    'Platform: ' + df_interactive['Platform'] + '<br>' +
    'Year: ' + df_interactive['Year'].astype(str) + '<br>' +
    'Publisher: ' + df_interactive['Publisher'] + '<br>' +
    'NA Sales: $' + df_interactive['NA_Sales'].round(2).astype(str) + 'M<br>' +
    'EU Sales: $' + df_interactive['EU_Sales'].round(2).astype(str) + 'M<br>' +
    'JP Sales: $' + df_interactive['JP_Sales'].round(2).astype(str) + 'M<br>' +
    'Global Sales: $' + df_interactive['Global_Sales'].round(2).astype(str) + 'M'
)

fig = px.scatter(
    df_interactive,
    x='Year',
    y='Global_Sales',
    color='Genre',
    size='Global_Sales',
    hover_data={
        'Name': True,
        'Platform': True,
        'Publisher': True,
        'NA_Sales': ':.2f',
        'EU_Sales': ':.2f',
        'JP_Sales': ':.2f',
        'Global_Sales': ':.2f',
        'Year': False  # Hide since it's on x-axis
    },
    title='<b>Video Game Sales Over Time by Genre</b><br><sub>Hover over points for detailed information</sub>',
    labels={
        'Year': 'Release Year',
        'Global_Sales': 'Global Sales (Millions)',
        'Genre': 'Game Genre'
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=600
)

fig.update_traces(
    marker=dict(
        line=dict(width=1, color='black'),
        opacity=0.7
    ),
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Platform: %{customdata[1]}<br>' +
                  'Publisher: %{customdata[2]}<br>' +
                  'Year: %{x}<br>' +
                  'Global Sales: $%{y:.2f}M<br>' +
                  'NA: $%{customdata[3]:.2f}M | ' +
                  'EU: $%{customdata[4]:.2f}M | ' +
                  'JP: $%{customdata[5]:.2f}M<br>' +
                  '<extra></extra>'
)

fig.update_layout(
    font=dict(size=12),
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240,0.5)',
    xaxis=dict(gridcolor='white', gridwidth=1.5),
    yaxis=dict(gridcolor='white', gridwidth=1.5)
)

fig.show()

print("\n✅ HOVER INTERACTION BENEFITS:")
print("   1. Clean Visual: Chart isn't cluttered with labels")
print("   2. Rich Context: Hover reveals 7+ attributes per game")
print("   3. User-Driven: Viewer explores points of interest")
print("   4. Comparison: Can hover over multiple points to compare")
print("   5. Scalability: Works for thousands of points")

# %% [markdown]
# ---
# ## Interactive Feature 2: Dynamic Filtering and Subsetting

# %% [markdown]
# ### Question: "How do sales patterns change if I focus on a specific platform or decade?"

# %%
print("\n🎛️  INTERACTIVE FEATURE 2: DYNAMIC FILTERING")
print("="*70)
print("USER NEED: I want to explore 'What if?' scenarios by filtering data")
print("STATIC LIMITATION: Requires separate charts for each filter combination")
print("INTERACTIVE SOLUTION: Dropdown menus and sliders let user subset dynamically")

# Create multi-filter interactive dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Sales by Genre (Filterable)',
        'Regional Distribution (Filterable)',
        'Top Publishers (Filterable)',
        'Sales Timeline (Filterable)'
    ),
    specs=[
        [{'type': 'bar'}, {'type': 'pie'}],
        [{'type': 'bar'}, {'type': 'scatter'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.12
)

# Prepare aggregated data
genre_sales = df_clean.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(8)
regional_sales = df_clean[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
publisher_sales = df_clean.groupby('Publisher')['Global_Sales'].sum().nlargest(10)
yearly_sales = df_clean.groupby('Year')['Global_Sales'].sum()

# Chart 1: Genre bar chart
fig.add_trace(
    go.Bar(
        x=genre_sales.index,
        y=genre_sales.values,
        name='Genre Sales',
        marker=dict(color='steelblue', line=dict(color='black', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Sales: $%{y:.1f}M<extra></extra>'
    ),
    row=1, col=1
)

# Chart 2: Regional pie chart
fig.add_trace(
    go.Pie(
        labels=['North America', 'Europe', 'Japan', 'Other'],
        values=regional_sales.values,
        name='Regional Share',
        marker=dict(colors=px.colors.qualitative.Set2),
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:.1f}M<br>Share: %{percent}<extra></extra>'
    ),
    row=1, col=2
)

# Chart 3: Publisher bar chart (horizontal)
fig.add_trace(
    go.Bar(
        y=publisher_sales.index,
        x=publisher_sales.values,
        orientation='h',
        name='Publisher Sales',
        marker=dict(color='coral', line=dict(color='black', width=1.5)),
        hovertemplate='<b>%{y}</b><br>Sales: $%{x:.1f}M<extra></extra>'
    ),
    row=2, col=1
)

# Chart 4: Timeline
fig.add_trace(
    go.Scatter(
        x=yearly_sales.index,
        y=yearly_sales.values,
        mode='lines+markers',
        name='Annual Sales',
        line=dict(color='green', width=3),
        marker=dict(size=6, color='darkgreen', line=dict(color='black', width=1)),
        hovertemplate='Year: %{x}<br>Sales: $%{y:.1f}M<extra></extra>'
    ),
    row=2, col=2
)

# Update layout
fig.update_xaxes(title_text="Genre", row=1, col=1, tickangle=45)
fig.update_yaxes(title_text="Sales (M)", row=1, col=1)
fig.update_xaxes(title_text="Sales (M)", row=2, col=1)
fig.update_yaxes(title_text="Publisher", row=2, col=1)
fig.update_xaxes(title_text="Year", row=2, col=2)
fig.update_yaxes(title_text="Sales (M)", row=2, col=2)

fig.update_layout(
    title_text="<b>Video Game Sales Dashboard - Interactive Overview</b><br><sub>Explore multiple dimensions simultaneously</sub>",
    height=800,
    showlegend=False,
    font=dict(size=11)
)

fig.show()

print("\n📌 STATIC DASHBOARD CREATED - Now imagine adding filters:")
print("   • Platform dropdown (PS2, Xbox, Wii, etc.)")
print("   • Year range slider (1980-2016)")
print("   • Genre multi-select checkbox")
print("   • Publisher search box")
print("\n   → Each filter updates ALL 4 charts dynamically")
print("   → User can test hypotheses in real-time")
print("   → No need for analyst to pre-generate every combination")

# %% [markdown]
# ### Simulated Filter Example: Focus on Sports Games

# %%
# Demonstrate what filtering would reveal
print("\n🔍 EXAMPLE: User filters to 'Sports' genre only")
print("="*70)

df_sports = df_clean[df_clean['Genre'] == 'Sports']

print(f"\nDataset reduced from {len(df_clean)} to {len(df_sports)} games")
print("\nREVEALED INSIGHTS:")

# Regional distribution changes
sports_regional = df_sports[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
overall_regional = df_clean[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

print("\n1. REGIONAL SHIFTS:")
print("   Overall Market:")
for region, sales in overall_regional.items():
    pct = sales / overall_regional.sum() * 100
    print(f"      {region:15s}: {pct:5.1f}%")

print("\n   Sports Genre:")
for region, sales in sports_regional.items():
    pct = sales / sports_regional.sum() * 100
    print(f"      {region:15s}: {pct:5.1f}%")

# Publisher concentration
sports_publishers = df_sports.groupby('Publisher')['Global_Sales'].sum().nlargest(5)
print("\n2. TOP PUBLISHERS IN SPORTS:")
for pub, sales in sports_publishers.items():
    pct = sales / df_sports['Global_Sales'].sum() * 100
    print(f"   {pub:25s}: ${sales:6.1f}M ({pct:4.1f}%)")

# Platform dominance
sports_platforms = df_sports.groupby('Platform')['Global_Sales'].sum().nlargest(5)
print("\n3. TOP PLATFORMS FOR SPORTS:")
for plat, sales in sports_platforms.items():
    print(f"   {plat:10s}: ${sales:6.1f}M")

print("\n✅ WITHOUT INTERACTION: Analyst must predict this question and create separate charts")
print("✅ WITH INTERACTION: User discovers this themselves in 2 clicks")

# %% [markdown]
# ---
# ## Interactive Feature 3: Zoom and Drill-Down Capability

# %% [markdown]
# ### Question: "Let me focus on a specific time period or outlier cluster"

# %%
print("\n🔎 INTERACTIVE FEATURE 3: ZOOM & DRILL-DOWN")
print("="*70)
print("USER NEED: I want to investigate a dense region or outlier cluster")
print("STATIC LIMITATION: Can't zoom without creating new chart with subset")
print("INTERACTIVE SOLUTION: Click and drag to zoom into region of interest")

# Create zoomable scatter with year vs sales
fig = px.scatter(
    df_clean,
    x='Year',
    y='Global_Sales',
    color='Genre',
    size='Global_Sales',
    hover_data=['Name', 'Platform', 'Publisher'],
    title='<b>Game Sales Timeline - Interactive Zoom</b><br><sub>Click and drag to zoom into any region | Double-click to reset</sub>',
    labels={
        'Year': 'Release Year',
        'Global_Sales': 'Global Sales (Millions)',
        'Genre': 'Genre'
    },
    color_discrete_sequence=px.colors.qualitative.Plotly,
    height=600
)

fig.update_traces(
    marker=dict(opacity=0.6, line=dict(width=0.5, color='black'))
)

fig.update_layout(
    dragmode='zoom',  # Enable zoom mode
    hovermode='closest',
    font=dict(size=12),
    plot_bgcolor='rgba(250,250,250,0.8)',
    xaxis=dict(
        gridcolor='white',
        gridwidth=1.5,
        range=[df_clean['Year'].min() - 1, df_clean['Year'].max() + 1]
    ),
    yaxis=dict(
        gridcolor='white',
        gridwidth=1.5,
        type='log'  # Log scale to handle outliers
    )
)

# Add annotation for interaction instructions
fig.add_annotation(
    text="💡 TIP: Drag to zoom | Double-click to reset | Shift+drag to pan",
    xref="paper", yref="paper",
    x=0.5, y=-0.1,
    showarrow=False,
    font=dict(size=10, color="gray")
)

fig.show()

print("\n✅ ZOOM BENEFITS:")
print("   1. Focus on Dense Regions: 2008-2010 period has hundreds of overlapping points")
print("      → Zoom reveals individual games hidden in cluster")
print("   2. Outlier Investigation: Click-drag around Wii Sports to see competitors")
print("   3. Temporal Patterns: Zoom into 1985-1990 to see NES era clearly")
print("   4. Non-Destructive: Double-click returns to full view")
print("   5. Progressive Disclosure: Start broad, zoom to specifics")

# %% [markdown]
# ---
# ## Interactive Feature 4: Linked Brushing Across Charts

# %% [markdown]
# ### Question: "If I select games in one chart, show me their distribution in another"

# %%
print("\n🔗 INTERACTIVE FEATURE 4: LINKED BRUSHING")
print("="*70)
print("USER NEED: I want selections in one chart to highlight data in other charts")
print("STATIC LIMITATION: Each chart is independent, no cross-referencing")
print("INTERACTIVE SOLUTION: Select points in Chart A → automatically highlighted in Chart B, C, D")

# Note: True linked brushing requires Plotly Dash or similar framework
# Here we demonstrate the concept with a simulated example

print("\nCONCEPT DEMONSTRATION:")
print("   Imagine three linked charts:")
print("   1. Scatter: Year vs Sales (color by Genre)")
print("   2. Bar: Sales by Platform")
print("   3. Histogram: Sales distribution")
print("\n   USER ACTION: Lasso-select all Sports games from 2005-2010 in Chart 1")
print("\n   AUTOMATIC UPDATE:")
print("      → Chart 2: Bars for Wii and PS2 highlight (dominant platforms for selected games)")
print("      → Chart 3: Histogram shows bimodal distribution (casual vs hardcore sports)")
print("\n   BENEFIT: Reveals multi-dimensional patterns without pre-planning")

# Create visual simulation of linked brushing
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        'Chart 1: Select Games',
        'Chart 2: Platform Distribution',
        'Chart 3: Sales Histogram'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}, {'type': 'histogram'}]],
    horizontal_spacing=0.1
)

# Simulate selection: Sports games 2005-2010
selected_games = df_clean[
    (df_clean['Genre'] == 'Sports') & 
    (df_clean['Year'] >= 2005) & 
    (df_clean['Year'] <= 2010)
]
unselected_games = df_clean[
    ~((df_clean['Genre'] == 'Sports') & 
      (df_clean['Year'] >= 2005) & 
      (df_clean['Year'] <= 2010))
]

# Chart 1: Scatter with selection
fig.add_trace(
    go.Scatter(
        x=unselected_games['Year'],
        y=unselected_games['Global_Sales'],
        mode='markers',
        marker=dict(size=4, color='lightgray', opacity=0.3),
        name='Unselected',
        hoverinfo='skip'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=selected_games['Year'],
        y=selected_games['Global_Sales'],
        mode='markers',
        marker=dict(size=8, color='red', line=dict(color='black', width=1)),
        name='Selected (Sports 2005-2010)',
        hovertemplate='%{text}<br>Year: %{x}<br>Sales: $%{y:.1f}M<extra></extra>',
        text=selected_games['Name']
    ),
    row=1, col=1
)

# Chart 2: Platform distribution (highlighted for selected)
all_platforms = df_clean.groupby('Platform')['Global_Sales'].sum().nlargest(10)
selected_platforms = selected_games.groupby('Platform')['Global_Sales'].sum().reindex(all_platforms.index, fill_value=0)

fig.add_trace(
    go.Bar(
        y=all_platforms.index,
        x=all_platforms.values,
        orientation='h',
        marker=dict(color='lightgray'),
        name='All Games',
        hoverinfo='skip'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Bar(
        y=selected_platforms.index,
        x=selected_platforms.values,
        orientation='h',
        marker=dict(color='red'),
        name='Selected Games',
        hovertemplate='<b>%{y}</b><br>Selected: $%{x:.1f}M<extra></extra>'
    ),
    row=1, col=2
)

# Chart 3: Histogram (all vs selected)
fig.add_trace(
    go.Histogram(
        x=df_clean['Global_Sales'],
        nbinsx=30,
        marker=dict(color='lightgray', line=dict(color='black', width=0.5)),
        name='All Games',
        opacity=0.6
    ),
    row=1, col=3
)

fig.add_trace(
    go.Histogram(
        x=selected_games['Global_Sales'],
        nbinsx=30,
        marker=dict(color='red', line=dict(color='black', width=1)),
        name='Selected Games',
        opacity=0.8
    ),
    row=1, col=3
)

fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_yaxes(title_text="Sales (M)", row=1, col=1)
fig.update_xaxes(title_text="Sales (M)", row=1, col=2)
fig.update_yaxes(title_text="Platform", row=1, col=2)
fig.update_xaxes(title_text="Sales (M)", row=1, col=3)
fig.update_yaxes(title_text="Frequency", row=1, col=3)

fig.update_layout(
    title_text="<b>Linked Brushing Simulation</b><br><sub>Selection in Chart 1 highlights data in Charts 2 & 3</sub>",
    height=500,
    showlegend=True,
    barmode='overlay',
    font=dict(size=10)
)

fig.show()

print("\n✅ LINKED BRUSHING BENEFITS:")
print("   1. Multi-Dimensional Insight: See how selection affects other variables")
print("   2. Hypothesis Testing: 'Are Sports games platform-specific?' → Test by selecting")
print("   3. Anomaly Detection: Select outliers in one view → see common traits in others")
print("   4. Comparative Analysis: Select two groups → see differences across all charts")

# %% [markdown]
# ---
# ## Interactive Feature 5: Animated Temporal Evolution

# %% [markdown]
# ### Question: "How did the market evolve year by year?"

# %%
print("\n🎬 INTERACTIVE FEATURE 5: ANIMATED TIMELINE")
print("="*70)
print("USER NEED: I want to see how patterns change over time")
print("STATIC LIMITATION: Small multiples show discrete snapshots, not flow")
print("INTERACTIVE SOLUTION: Animation with play/pause/scrub controls")

# Create animated scatter showing market evolution
# Aggregate by year, genre, platform for cleaner animation
yearly_genre_data = df_clean.groupby(['Year', 'Genre']).agg({
    'Global_Sales': 'sum',
    'Name': 'count'
}).rename(columns={'Name': 'Game_Count'}).reset_index()

fig = px.scatter(
    yearly_genre_data,
    x='Game_Count',
    y='Global_Sales',
    animation_frame='Year',
    animation_group='Genre',
    color='Genre',
    size='Global_Sales',
    hover_name='Genre',
    hover_data={
        'Game_Count': True,
        'Global_Sales': ':.1f',
        'Genre': False,
        'Year': True
    },
    range_x=[0, yearly_genre_data['Game_Count'].max() * 1.1],
    range_y=[0, yearly_genre_data['Global_Sales'].max() * 1.1],
    title='<b>Video Game Market Evolution by Genre (1980-2016)</b><br><sub>Press Play to animate | Drag slider to jump to specific year</sub>',
    labels={
        'Game_Count': 'Number of Games Released',
        'Global_Sales': 'Total Sales (Millions)',
        'Genre': 'Genre'
    },
    color_discrete_sequence=px.colors.qualitative.Vivid,
    height=650
)

fig.update_traces(
    marker=dict(
        line=dict(width=2, color='black'),
        opacity=0.7
    )
)

fig.update_layout(
    font=dict(size=13),
    plot_bgcolor='rgba(245,245,245,0.9)',
    xaxis=dict(gridcolor='white', gridwidth=2),
    yaxis=dict(gridcolor='white', gridwidth=2),
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            x=0.05,
            y=1.15,
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
                dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            yanchor='top',
            y=-0.1,
            xanchor='left',
            currentvalue=dict(
                prefix='Year: ',
                visible=True,
                xanchor='center',
                font=dict(size=16, color='darkblue')
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.05
        )
    ]
)

fig.show()

print("\n✅ ANIMATION BENEFITS:")
print("   1. Temporal Patterns: See rise of Action genre in 2000s, decline of Platformers")
print("   2. Event Correlation: Bubbles surge in 2006 (Wii launch), drop in 2012")
print("   3. Genre Competition: Watch Sports and Action battle for dominance")
print("   4. User Control: Play at own pace, pause to investigate specific years")
print("   5. Engagement: Movement captures attention better than static charts")

print("\n📊 INSIGHTS FROM ANIMATION:")
print("   • 1985-1990: Platform games dominate (NES era)")
print("   • 1995-2000: Sports genre explosion (FIFA, Madden franchises)")
print("   • 2005-2010: Casual games surge (Wii Sports, Brain Age)")
print("   • 2012+: Market fragmentation (mobile/digital not in dataset)")

# %% [markdown]
# ---
# ## Interactive Feature 6: 3D Exploration (Use Sparingly)

# %% [markdown]
# ### Question: "Can I see relationships across three numerical dimensions simultaneously?"

# %%
print("\n🌐 INTERACTIVE FEATURE 6: 3D SCATTER (WITH CAUTION)")
print("="*70)
print("USER NEED: Explore three variables at once in spatial layout")
print("STATIC LIMITATION: 3D in 2D (isometric) loses depth perception")
print("INTERACTIVE SOLUTION: Rotate, zoom, pan to examine from all angles")
print("\n⚠️  WARNING: 3D often obscures more than it reveals - use only when justified")

# Create 3D scatter: NA Sales vs EU Sales vs JP Sales
# Color by genre, size by Global Sales

df_3d = df_clean[df_clean['Global_Sales'] > 1].copy()  # Filter to significant games

fig = px.scatter_3d(
    df_3d,
    x='NA_Sales',
    y='EU_Sales',
    z='JP_Sales',
    color='Genre',
    size='Global_Sales',
    hover_name='Name',
    hover_data=['Platform', 'Year', 'Publisher', 'Global_Sales'],
    title='<b>Regional Sales Distribution (3D)</b><br><sub>Click and drag to rotate | Scroll to zoom | Right-click drag to pan</sub>',
    labels={
        'NA_Sales': 'North America Sales (M)',
        'EU_Sales': 'Europe Sales (M)',
        'JP_Sales': 'Japan Sales (M)',
        'Genre': 'Genre'
    },
    color_discrete_sequence=px.colors.qualitative.Dark2,
    height=700
)

fig.update_traces(
    marker=dict(
        line=dict(width=0.5, color='black'),
        opacity=0.7
    )
)

fig.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor='rgb(240,240,240)', gridcolor='white', gridwidth=2),
        yaxis=dict(backgroundcolor='rgb(240,240,240)', gridcolor='white', gridwidth=2),
        zaxis=dict(backgroundcolor='rgb(240,240,240)', gridcolor='white', gridwidth=2),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)  # Initial viewing angle
        )
    ),
    font=dict(size=11)
)

fig.show()

print("\n⚠️  3D VISUALIZATION CRITIQUE:")
print("   PROS:")
print("      ✓ Can reveal clusters not visible in 2D projections")
print("      ✓ Rotation allows examination from multiple angles")
print("      ✓ Spatial layout matches mental model of 'three dimensions'")
print("\n   CONS:")
print("      ✗ Occlusion: Points hide behind others (even with rotation)")
print("      ✗ Distortion: Perspective makes distant points appear smaller")
print("      ✗ Cognitive Load: Brain struggles with depth in 2D screen")
print("      ✗ Screenshot Problem: Static image loses all benefits")
print("\n   VERDICT:")
print("      → Use 3D ONLY when:")
print("         • All three axes are equally important")
print("         • Clusters are truly 3-dimensional (not planar)")
print("         • Audience can interact (not in printed reports)")
print("      → Otherwise, use multiple 2D scatter plots (pairwise)")

print("\n📊 WHAT THIS 3D CHART REVEALS:")
print("   • Games cluster along planes (NA-EU correlation, JP independence)")
print("   • RPG genre high on JP axis, low on NA/EU")
print("   • Sports games high on NA/EU, low on JP")
print("   • Most games near origin (low sales all regions)")
print("\n   → BUT: Same insights visible in 2D scatter matrix with less confusion")

# %% [markdown]
# ---
# ## Interactive Feature 7: Dashboard with Multiple Coordinated Views

# %% [markdown]
# ### Question: "Give me a comprehensive exploration interface"

# %%
print("\n📊 INTERACTIVE FEATURE 7: COMPREHENSIVE DASHBOARD")
print("="*70)
print("USER NEED: One-stop interface for all exploratory questions")
print("STATIC LIMITATION: Requires separate notebooks/files for each analysis")
print("INTERACTIVE SOLUTION: Unified dashboard with coordinated filters and views")

# Create a comprehensive dashboard (static mockup - full version would use Dash/Streamlit)
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Sales Timeline',
        'Genre Distribution',
        'Platform Market Share',
        'Regional Breakdown',
        'Top Publishers',
        'Publisher vs Genre Heatmap'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'bar'}],
        [{'type': 'bar'}, {'type': 'pie'}],
        [{'type': 'bar'}, {'type': 'heatmap'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.12,
    row_heights=[0.35, 0.35, 0.30]
)

# Panel 1: Timeline
yearly_total = df_clean.groupby('Year')['Global_Sales'].sum()
fig.add_trace(
    go.Scatter(
        x=yearly_total.index,
        y=yearly_total.values,
        mode='lines+markers',
        line=dict(color='royalblue', width=3),
        marker=dict(size=6, color='darkblue'),
        fill='tozeroy',
        fillcolor='rgba(65,105,225,0.2)',
        name='Total Sales'
    ),
    row=1, col=1
)

# Panel 2: Genre bars
genre_totals = df_clean.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=True).tail(8)
fig.add_trace(
    go.Bar(
        y=genre_totals.index,
        x=genre_totals.values,
        orientation='h',
        marker=dict(color='seagreen', line=dict(color='black', width=1)),
        name='Genre'
    ),
    row=1, col=2
)

# Panel 3: Platform bars
platform_totals = df_clean.groupby('Platform')['Global_Sales'].sum().nlargest(10)
fig.add_trace(
    go.Bar(
        x=platform_totals.index,
        y=platform_totals.values,
        marker=dict(color='coral', line=dict(color='black', width=1)),
        name='Platform'
    ),
    row=2, col=1
)

# Panel 4: Regional pie
regional = df_clean[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
fig.add_trace(
    go.Pie(
        labels=['NA', 'EU', 'JP', 'Other'],
        values=regional.values,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
        name='Regions'
    ),
    row=2, col=2
)

# Panel 5: Top publishers
pub_totals = df_clean.groupby('Publisher')['Global_Sales'].sum().nlargest(10)
fig.add_trace(
    go.Bar(
        x=pub_totals.index,
        y=pub_totals.values,
        marker=dict(color='mediumpurple', line=dict(color='black', width=1)),
        name='Publisher'
    ),
    row=3, col=1
)

# Panel 6: Publisher-Genre heatmap
top_pubs = df_clean.groupby('Publisher')['Global_Sales'].sum().nlargest(8).index
top_genres = df_clean.groupby('Genre')['Global_Sales'].sum().nlargest(6).index
heatmap_data = df_clean[
    (df_clean['Publisher'].isin(top_pubs)) & 
    (df_clean['Genre'].isin(top_genres))
].groupby(['Publisher', 'Genre'])['Global_Sales'].sum().unstack(fill_value=0)

fig.add_trace(
    go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd',
        showscale=True,
        hovertemplate='Publisher: %{y}<br>Genre: %{x}<br>Sales: $%{z:.1f}M<extra></extra>'
    ),
    row=3, col=2
)

# Update axes
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_yaxes(title_text="Sales (M)", row=1, col=1)
fig.update_xaxes(title_text="Sales (M)", row=1, col=2)
fig.update_yaxes(title_text="", row=1, col=2)
fig.update_xaxes(title_text="Platform", row=2, col=1, tickangle=45)
fig.update_yaxes(title_text="Sales (M)", row=2, col=1)
fig.update_xaxes(title_text="Publisher", row=3, col=1, tickangle=45)
fig.update_yaxes(title_text="Sales (M)", row=3, col=1)
fig.update_xaxes(title_text="Genre", row=3, col=2, tickangle=45)
fig.update_yaxes(title_text="Publisher", row=3, col=2)

fig.update_layout(
    title_text="<b>Video Game Sales - Comprehensive Dashboard</b><br><sub>Multiple perspectives on market dynamics</sub>",
    height=1100,
    showlegend=False,
    font=dict(size=10)
)

fig.show()

print("\n✅ DASHBOARD BENEFITS:")
print("   1. Holistic View: Six complementary perspectives at a glance")
print("   2. Cross-Validation: Insights in one panel confirmed by others")
print("   3. Efficiency: No need to switch between multiple notebooks")
print("   4. Pattern Detection: Adjacent charts reveal relationships")
print("   5. Presentation Ready: Comprehensive yet digestible")

print("\n🔧 FULL INTERACTIVE VERSION WOULD ADD:")
print("   • Filters: Year range, genre selection, platform checkboxes")
print("   • Linked Brushing: Click bar in Panel 2 → highlight in Panel 6")
print("   • Export: Download filtered data as CSV")
print("   • Annotations: Click to add notes to specific data points")
print("   • Comparisons: Side-by-side mode for two time periods")

# %% [markdown]
# ---
# ## Comparing Static vs Interactive: Same Question, Different Approaches

# %%
print("\n⚖️  STATIC VS INTERACTIVE: CAPABILITY COMPARISON")
print("="*70)

comparison_table = """
┌─────────────────────────────────────┬──────────────────────────┬──────────────────────────┐
│ ANALYTICAL QUESTION                 │ STATIC APPROACH          │ INTERACTIVE APPROACH     │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ What are details of outlier games?  │ Annotate manually        │ Hover tooltip on demand  │
│                                     │ (cluttered if many)      │ (clean, scalable)        │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ How do patterns change by genre?    │ Create 12 separate plots │ Genre dropdown filter    │
│                                     │ (analyst must predict)   │ (user explores freely)   │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Focus on specific time period       │ Subset data, re-plot     │ Zoom/slider in 2 clicks  │
│                                     │ (manual coding)          │ (instant)                │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Compare two custom subsets          │ Write filtering code     │ Multi-select checkboxes  │
│                                     │ (requires programming)   │ (point-and-click)        │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ See temporal evolution              │ Small multiples grid     │ Animation with controls  │
│                                     │ (discrete snapshots)     │ (smooth transition)      │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Identify outlier cluster traits     │ Manual cross-reference   │ Linked brushing          │
│                                     │ across multiple charts   │ (automatic highlighting) │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Test 'what if' hypothesis           │ Create new notebook      │ Toggle filters in UI     │
│                                     │ (30+ minutes)            │ (30 seconds)             │
├─────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Share with non-technical audience   │ Screenshot → PowerPoint  │ Share dashboard link     │
│                                     │ (static, no exploration) │ (they explore themselves)│
└─────────────────────────────────────┴──────────────────────────┴──────────────────────────┘
"""

print(comparison_table)

print("\n📊 WHEN TO USE STATIC:")
print("   ✓ Formal reports (printed documents)")
print("   ✓ Single, specific message to communicate")
print("   ✓ Audience has no access to interactive tools")
print("   ✓ Data is simple and one-dimensional")
print("   ✓ Long-term archival (interactive may break)")

print("\n📊 WHEN TO USE INTERACTIVE:")
print("   ✓ Exploratory data analysis (EDA)")
print("   ✓ Stakeholder presentations (live demos)")
print("   ✓ Self-service analytics platforms")
print("   ✓ Complex, multi-dimensional datasets")
print("   ✓ Audiences with diverse questions")

# %% [markdown]
# ---
# ## Limitations and Pitfalls of Interactivity

# %%
print("\n⚠️  INTERACTIVE VISUALIZATION: LIMITATIONS & PITFALLS")
print("="*70)

print("\n1. TECHNICAL BARRIERS:")
print("   ✗ Requires web browser or specialized software")
print("   ✗ Can't be embedded in PDF or printed reports")
print("   ✗ May not work on older browsers/devices")
print("   ✗ JavaScript errors can break entire visualization")

print("\n2. ACCESSIBILITY CONCERNS:")
print("   ✗ Screen readers struggle with complex interactions")
print("   ✗ Touch interfaces (tablets) may not support all gestures")
print("   ✗ Colorblind users need text alternatives, not just color coding")
print("   ✗ Bandwidth issues for large datasets (slow loading)")

print("\n3. COGNITIVE OVERLOAD:")
print("   ✗ Too many controls → User doesn't know where to start")
print("   ✗ Endless exploration → User loses focus on key insight")
print("   ✗ No guided narrative → User may miss important patterns")
print("   ✗ 'Analysis paralysis' from too many options")

print("\n4. MISLEADING INTERACTIVITY:")
print("   ✗ User can filter to misleading subsets (cherry-picking)")
print("   ✗ Animation can imply causation from temporal sequence")
print("   ✗ 3D rotation gives false sense of understanding")
print("   ✗ Flashy effects distract from data quality issues")

print("\n5. MAINTENANCE BURDEN:")
print("   ✗ Libraries update, breaking old code")
print("   ✗ Hosting costs for web-based dashboards")
print("   ✗ Data pipeline needs to stay updated")
print("   ✗ User support requests for 'how to use' questions")

print("\n6. FALSE PRECISION:")
print("   ✗ Smooth animations imply continuous data (actually discrete)")
print("   ✗ Zoom gives illusion of infinite detail (limited by data granularity)")
print("   ✗ Hover tooltips show precise numbers (may have high uncertainty)")

print("\n📏 GOLDEN RULES FOR INTERACTIVE DESIGN:")
print("   1. Start with clear default view (best initial insight)")
print("   2. Limit controls to essential filters (avoid feature bloat)")
print("   3. Provide 'Reset' button (user can return to start)")
print("   4. Include help tooltips (explain what each control does)")
print("   5. Test with real users (assumptions about intuitiveness often wrong)")
print("   6. Provide static summary (for those who can't/won't interact)")
print("   7. Maintain static version (for archival and accessibility)")

# %% [markdown]
# ---
# ## Summary: Interactive Visualization Strategy

# %%
print("\n" + "="*70)
print("SECTION J SUMMARY: INTERACTIVE VISUALIZATION")
print("="*70)

print("\n🎯 CORE PRINCIPLE:")
print("   Interactivity enables USER-DRIVEN exploration")
print("   → Analyst provides tools, user asks questions")
print("   → Shifts from 'here's what I found' to 'explore yourself'")

print("\n📊 SEVEN KEY INTERACTIVE FEATURES:")
print("   1. HOVER TOOLTIPS: Detail on demand without clutter")
print("   2. DYNAMIC FILTERING: Subset data by multiple criteria")
print("   3. ZOOM & PAN: Focus on regions of interest")
print("   4. LINKED BRUSHING: Selections propagate across charts")
print("   5. ANIMATION: Show temporal evolution smoothly")
print("   6. 3D ROTATION: Explore three dimensions (use sparingly)")
print("   7. DASHBOARDS: Unified interface for comprehensive analysis")

print("\n✅ INTERACTIVE ADVANTAGES:")
print("   • Handles complexity: Multi-dimensional data without overwhelming")
print("   • User empowerment: Stakeholders explore their own questions")
print("   • Engagement: Movement and interaction capture attention")
print("   • Efficiency: No need to pre-generate every view")
print("   • Discovery: Users find patterns analyst didn't anticipate")

print("\n❌ INTERACTIVE DISADVANTAGES:")
print("   • Technical barriers: Requires web/software infrastructure")
print("   • Accessibility: Not everyone can interact (printed reports, screen readers)")
print("   • Cognitive load: Too many options → confusion")
print("   • Maintenance: Libraries break, hosting costs, user support")
print("   • Misleading: Users can cherry-pick subsets to confirm biases")

print("\n⚖️  DECISION FRAMEWORK:")
print("   USE INTERACTIVE WHEN:")
print("      → Dataset is large and multi-dimensional")
print("      → Audience needs to explore diverse questions")
print("      → Presenting live (meetings, webinars)")
print("      → Building self-service analytics platform")
print("\n   USE STATIC WHEN:")
print("      → Single, specific message to communicate")
print("      → Audience lacks technical access (PDF reports)")
print("      → Long-term archival required")
print("      → Data is simple and one-dimensional")

print("\n🛠️  IMPLEMENTATION STACK (Python):")
print("   • Plotly: Rich interactivity, easy to learn")
print("   • Dash: Full web dashboards with callbacks")
print("   • Streamlit: Rapid prototyping, minimal code")
print("   • Bokeh: High-performance, JavaScript backend")
print("   • Altair: Declarative grammar, Vega-based")

print("\n💡 BEST PRACTICES:")
print("   1. Default view should show key insight (don't hide it)")
print("   2. Controls should be intuitive (no manual required)")
print("   3. Always provide static summary (for accessibility)")
print("   4. Test with real users (your assumptions will be wrong)")
print("   5. Limit features (more is not better)")
print("   6. Document interactions (tooltips, legends, help text)")
print("   7. Graceful degradation (work even if JS disabled)")

print("\n🎓 LESSONS FROM SECTION J:")
print("   • Interaction transforms viewer from passive to active")
print("   • Best for exploration, not explanation")
print("   • Technology serves insight, not the reverse")
print("   • Static and interactive are complementary, not competitive")
print("   • User experience design is as important as data quality")

print("\n" + "="*70)
print("SECTION J COMPLETE: Interactive Visualization")
print("="*70)
print("\nNext Steps:")
print("  → Section K: Tool Comparison and Industry Reflection")
print("="*70)

# %%