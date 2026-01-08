# %% [markdown]
# # Data Analysis: A Comprehensive Self-Study Guide
# This notebook covers essential data analysis skills with the Iris dataset
#
# ## Table of Contents
# 1. Introduction to Data Analysis
# 2. Data Importing Methods
# 3. Understanding Data Distribution
# 4. Data Plotting Techniques and Best Practices
# 5. Stratified Sampling
# 6. Multiclass Classification
# 7. Advanced Data Visualization
# 8. Summary and Best Practices

# %% [markdown]
# # 1. Introduction to Data Analysis
#
# Data analysis is the process of inspecting, cleaning, transforming, and modeling data
# to discover useful information, draw conclusions, and support decision-making.
#
# ## Why These Skills Matter:
# - **Data Distribution**: Understanding how data is spread helps identify patterns
# - **Visualization**: Makes complex data understandable at a glance
# - **Sampling**: Ensures we work with representative data subsets
# - **Classification**: Enables predictions and pattern recognition
#
# ## Dataset: Iris Flower Dataset
# We'll use the famous Iris dataset, which contains measurements of 150 iris flowers
# from three different species. It's perfect for learning because:
# - Small enough to understand completely
# - Real-world biological data
# - Contains multiple classes (3 species)
# - Well-documented and widely used

# %%
# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For advanced statistical visualizations
from sklearn.datasets import load_iris  # To load the Iris dataset
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.tree import DecisionTreeClassifier  # For classification
from sklearn.metrics import classification_report, confusion_matrix  # For evaluation
import warnings  # To handle warnings

# Configure visualization settings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
plt.style.use('seaborn-v0_8-darkgrid')  # Set a professional plot style
sns.set_palette("husl")  # Use a colorful palette

print("✓ All libraries imported successfully!")

# %% [markdown]
# # 2. Data Importing Methods
#
# There are multiple ways to import data in Python. Let's explore the most common:
#
# ## Method 1: From sklearn datasets (built-in)

# %%
# Load the Iris dataset from sklearn
iris = load_iris()

# Understanding the structure of sklearn datasets
print("Dataset structure:")
print(f"- Keys: {iris.keys()}")
print(f"- Feature names: {iris.feature_names}")
print(f"- Target names: {iris.target_names}")
print(f"- Data shape: {iris.data.shape}")
print(f"- Target shape: {iris.target.shape}")

# %% [markdown]
# ## Method 2: Creating a Pandas DataFrame (recommended for analysis)

# %%
# Create a DataFrame with descriptive column names
df = pd.DataFrame(
    data=iris.data,  # The feature data (measurements)
    columns=iris.feature_names  # Column names
)

# Add the target variable (species) to the DataFrame
df['species'] = iris.target

# Map numeric targets to actual species names for better readability
df['species_name'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor', 
    2: 'virginica'
})

print("\n✓ Data successfully imported into DataFrame!")
print(f"DataFrame shape: {df.shape}")

# Display first few rows
df.head()

# %% [markdown]
# ## Method 3: Alternative - Reading from CSV file
# (This is how you would typically load your own data)
#
# ```python
# # If you had a CSV file:
# df = pd.read_csv('iris.csv')
#
# # Other common formats:
# df = pd.read_excel('data.xlsx')  # Excel files
# df = pd.read_json('data.json')   # JSON files
# df = pd.read_sql(query, connection)  # SQL databases
# ```

# %% [markdown]
# # 3. Understanding Data Distribution
#
# Data distribution tells us how values are spread across the dataset.
# Key concepts:
# - **Central Tendency**: Mean, median, mode
# - **Spread**: Range, variance, standard deviation
# - **Shape**: Skewness, kurtosis

# %%
print("="*70)
print("DATA DISTRIBUTION ANALYSIS")
print("="*70)

# 3.1 Basic Information
print("\n### Basic Dataset Information:")
df.info()

# %%
# 3.2 Statistical Summary
print("\n### Statistical Summary:")
df.describe()

# %% [markdown]
# ### Interpreting the Summary:
# - **count**: Number of non-null values
# - **mean**: Average value
# - **std**: Standard deviation (measure of spread)
# - **min/max**: Smallest and largest values
# - **25%, 50%, 75%**: Quartiles (divide data into 4 equal parts)

# %%
# 3.3 Distribution by Species
print("\n### Distribution of Species (Class Balance):")
species_counts = df['species_name'].value_counts()
print(species_counts)
print(f"\nDataset is balanced: {species_counts.std() < 1}")  # Check if classes are equal

# %%
# 3.4 Correlation Analysis
print("\n### Correlation Matrix:")
correlation_matrix = df.iloc[:, :-2].corr()  # Exclude species columns
print(correlation_matrix)

# %% [markdown]
# ### Understanding Correlation:
# - Values range from -1 to 1
# - Close to 1: Strong positive correlation (when one increases, other increases)
# - Close to -1: Strong negative correlation (when one increases, other decreases)
# - Close to 0: No linear relationship

# %% [markdown]
# # 4. Data Plotting Techniques and Best Practices
#
# Visualization is crucial for understanding data. Let's explore different plot types:
#
# ## Best Practices:
# 1. Choose the right plot for your data type
# 2. Always label axes clearly
# 3. Use titles that explain what you're showing
# 4. Include legends when comparing multiple items
# 5. Use appropriate color schemes
# 6. Avoid chart junk (unnecessary decorations)

# %%
# 4.1 Histogram - Shows distribution of a single variable
plt.figure(figsize=(15, 4))

for i, column in enumerate(df.columns[:-2], 1):
    plt.subplot(1, 4, i)
    plt.hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("\n✓ Histogram created: Shows frequency distribution of each feature")

# %%
# 4.2 Box Plot - Shows distribution, median, and outliers
plt.figure(figsize=(12, 6))
df_melted = df.melt(id_vars=['species_name'], 
                     value_vars=df.columns[:-2],
                     var_name='Measurement', 
                     value_name='Value')

sns.boxplot(data=df_melted, x='Measurement', y='Value', hue='species_name')
plt.title('Box Plot: Distribution of Features by Species', fontsize=14, fontweight='bold')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Value (cm)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
print("✓ Box plot created: Shows median, quartiles, and outliers")

# %%
# 4.3 Violin Plot - Combines box plot with distribution
plt.figure(figsize=(14, 6))
for i, column in enumerate(df.columns[:-2], 1):
    plt.subplot(1, 4, i)
    sns.violinplot(data=df, y=column, x='species_name', palette='Set2')
    plt.title(f'{column}')
    plt.xlabel('Species')
    plt.xticks(rotation=45)

plt.suptitle('Violin Plots: Detailed Distribution by Species', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
print("✓ Violin plot created: Shows distribution shape and density")

# %%
# 4.4 Scatter Plot - Shows relationship between two variables
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['sepal length (cm)'], 
                subset['sepal width (cm)'],
                label=species, 
                alpha=0.6, 
                s=100)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.title('Sepal Dimensions by Species', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['petal length (cm)'], 
                subset['petal width (cm)'],
                label=species, 
                alpha=0.6, 
                s=100)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.title('Petal Dimensions by Species', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Scatter plot created: Shows relationships between features")

# %%
# 4.5 Correlation Heatmap - Shows relationships between all variables
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center colormap at 0
            square=True,  # Make cells square
            linewidths=1,  # Add gridlines
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("✓ Heatmap created: Shows correlation strength between features")

# %%
# 4.6 Pair Plot - Shows all pairwise relationships
print("\nCreating pair plot (this may take a moment)...")
pairplot = sns.pairplot(df, 
                         hue='species_name',
                         diag_kind='kde',  # Kernel Density Estimate on diagonal
                         plot_kws={'alpha': 0.6},
                         height=2.5)
pairplot.fig.suptitle('Pair Plot: All Feature Relationships', y=1.02, fontsize=14)
plt.show()
print("✓ Pair plot created: Shows all possible feature combinations")

# %% [markdown]
# # 5. Implementing Stratified Sampling
#
# **What is Stratified Sampling?**
# Stratified sampling ensures that each class is proportionally represented in both
# training and testing sets. This is crucial for:
# - Maintaining class balance
# - Getting reliable model evaluation
# - Avoiding bias in predictions
#
# **Why it matters:**
# If we have 50 samples of each species (total 150), random sampling might give us:
# - Train: 40 setosa, 35 versicolor, 45 virginica
# - Test: 10 setosa, 15 versicolor, 5 virginica ❌ Unbalanced!
#
# With stratified sampling:
# - Train: 40 setosa, 40 versicolor, 40 virginica
# - Test: 10 setosa, 10 versicolor, 10 virginica ✓ Balanced!

# %%
print("="*70)
print("STRATIFIED SAMPLING DEMONSTRATION")
print("="*70)

# Prepare features (X) and target (y)
X = df.iloc[:, :-2]  # All columns except species columns
y = df['species']  # Target variable

# 5.1 Regular Random Split (for comparison)
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42  # For reproducibility
)

print("\n### Random Sampling (No Stratification):")
print("Training set distribution:")
print(pd.Series(y_train_random).value_counts().sort_index())
print("\nTest set distribution:")
print(pd.Series(y_test_random).value_counts().sort_index())

# %%
# 5.2 Stratified Split (recommended)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% for testing
    stratify=y,  # Maintain class proportions
    random_state=42
)

print("\n### Stratified Sampling (With Stratification):")
print("Training set distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nTest set distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# %%
# Visualize the difference
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random sampling visualization
axes[0].bar(['Setosa', 'Versicolor', 'Virginica'], 
            pd.Series(y_test_random).value_counts().sort_index(),
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Random Sampling\n(Unbalanced)', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, 15)

# Stratified sampling visualization
axes[1].bar(['Setosa', 'Versicolor', 'Virginica'], 
            pd.Series(y_test).value_counts().sort_index(),
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1].set_title('Stratified Sampling\n(Balanced)', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_ylim(0, 15)

plt.suptitle('Test Set Distribution Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n✓ Stratified sampling ensures balanced representation")

# %% [markdown]
# # 6. Working with Multiclass Classification
#
# **Binary vs Multiclass:**
# - Binary: 2 classes (Yes/No, True/False, Cat/Dog)
# - Multiclass: 3+ classes (Setosa/Versicolor/Virginica)
#
# **Why Multiclass is More Complex:**
# - Need to distinguish between multiple categories
# - Evaluation metrics are more nuanced
# - Requires careful handling of predictions

# %%
print("="*70)
print("MULTICLASS CLASSIFICATION")
print("="*70)

# Train a Decision Tree Classifier
# Decision Trees work by creating rules based on features
# Example: "If petal length < 2.5, then it's Setosa"
clf = DecisionTreeClassifier(
    max_depth=3,  # Limit tree depth to avoid overfitting
    random_state=42  # For reproducibility
)

# Fit the model (learn from training data)
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

print("\n✓ Model trained successfully!")
print(f"Training accuracy: {clf.score(X_train, y_train):.2%}")
print(f"Testing accuracy: {clf.score(X_test, y_test):.2%}")

# %%
# 6.1 Confusion Matrix - Shows actual vs predicted classes
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, 
            annot=True,  # Show numbers
            fmt='d',  # Integer format
            cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix: Multiclass Classification', fontsize=14, fontweight='bold')
plt.ylabel('Actual Species', fontsize=12)
plt.xlabel('Predicted Species', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Understanding the Confusion Matrix:
# - Diagonal values (top-left to bottom-right): Correct predictions
# - Off-diagonal values: Misclassifications
# - Perfect model would have all values on diagonal

# %%
# 6.2 Classification Report - Detailed metrics for each class
print("\n### Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# %% [markdown]
# ### Understanding Metrics:
# - **Precision**: Of all predicted as class X, how many were actually X?
#   (Precision = True Positives / (True Positives + False Positives))
#   
# - **Recall**: Of all actual class X, how many did we correctly identify?
#   (Recall = True Positives / (True Positives + False Negatives))
#   
# - **F1-Score**: Harmonic mean of precision and recall
#   (Good balance between the two)
#   
# - **Support**: Number of actual occurrences of each class in test set

# %%
# 6.3 Feature Importance - Which features matter most?
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
         color='skyblue', edgecolor='navy')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance in Classification', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n### Feature Importance:")
print(feature_importance)
print("\n✓ Petal features are most important for distinguishing species")

# %% [markdown]
# # 7. Advanced Data Visualization Techniques
#
# Beyond basic plots, let's create more sophisticated visualizations.

# %%
# 7.1 Distribution Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = df.columns[:-2]

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    
    for species in df['species_name'].unique():
        data = df[df['species_name'] == species][feature]
        ax.hist(data, alpha=0.5, label=species, bins=15, edgecolor='black')
    
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution: {feature}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions by Species (Overlaid)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n✓ Advanced distribution plots created")

# %%
# 7.2 Statistical Summary Visualization
summary_stats = df.groupby('species_name')[features].agg(['mean', 'std'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean values
summary_stats.xs('mean', axis=1, level=1).T.plot(kind='bar', ax=axes[0], 
                                                   width=0.8, edgecolor='black')
axes[0].set_title('Mean Values by Species', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Mean Value (cm)', fontsize=11)
axes[0].set_xlabel('Feature', fontsize=11)
axes[0].legend(title='Species')
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Standard deviation
summary_stats.xs('std', axis=1, level=1).T.plot(kind='bar', ax=axes[1], 
                                                 width=0.8, edgecolor='black')
axes[1].set_title('Standard Deviation by Species', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Standard Deviation', fontsize=11)
axes[1].set_xlabel('Feature', fontsize=11)
axes[1].legend(title='Species')
axes[1].grid(axis='y', alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
print("✓ Statistical summary visualizations created")

# %% [markdown]
# # 8. Summary and Best Practices
#
# ## Key Takeaways:
#
# ### 1. Data Importing:
# - Use pandas DataFrames for flexibility
# - Understand your data structure before analysis
# - Check data types and handle missing values
#
# ### 2. Understanding Distribution:
# - Always start with .describe() and .info()
# - Check for class balance in classification tasks
# - Look for outliers and anomalies
# - Understand correlations between features
#
# ### 3. Visualization Best Practices:
# - **Histograms**: For single variable distributions
# - **Box plots**: For comparing distributions and spotting outliers
# - **Scatter plots**: For relationships between two variables
# - **Heatmaps**: For correlations and confusion matrices
# - **Pair plots**: For comprehensive exploratory analysis
#
# ### 4. Stratified Sampling:
# - Always use stratification for classification problems
# - Ensures balanced representation in train/test splits
# - Leads to more reliable model evaluation
#
# ### 5. Multiclass Classification:
# - More complex than binary classification
# - Requires careful evaluation with confusion matrices
# - Use multiple metrics (precision, recall, F1-score)
# - Understand feature importance
#
# ## Next Steps for Practice:
# 1. Try different datasets from sklearn or Kaggle
# 2. Experiment with different classification algorithms
# 3. Create custom visualizations for your specific needs
# 4. Practice interpreting confusion matrices
# 5. Learn about cross-validation for more robust evaluation
#
# ## Common Pitfalls to Avoid:
# - ❌ Not checking for class imbalance
# - ❌ Using too many features without understanding them
# - ❌ Over-interpreting accuracy without checking other metrics
# - ❌ Not visualizing data before modeling
# - ❌ Forgetting to use stratification in train/test splits

# %%
print("\n" + "="*70)
print("TUTORIAL COMPLETE!")
print("="*70)
print("\n✓ You've learned:")
print("  • Data importing and DataFrame creation")
print("  • Statistical analysis and distribution understanding")
print("  • Multiple visualization techniques")
print("  • Stratified sampling implementation")
print("  • Multiclass classification with evaluation")
print("\nAll visualizations have been created. Review them carefully!")
print("\nHappy learning! 🎓📊")