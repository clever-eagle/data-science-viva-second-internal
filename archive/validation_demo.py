# %% [markdown]
# # Machine Learning Validation Techniques: A Complete Guide
#
# ## Why Validation Matters
#
# Imagine you're studying for an exam by only practicing with the same set of questions. You might memorize the answers perfectly, but fail when faced with new questions. This is exactly what happens when machine learning models **overfit** to training data.
#
# **Validation techniques** help us:
# - Estimate how well our model generalizes to unseen data
# - Detect overfitting early
# - Compare different models fairly
# - Tune hyperparameters without biasing our final evaluation
#
# In this notebook, we'll explore different validation strategies using the classic **Iris dataset** - a simple dataset perfect for understanding these concepts.

# %% [markdown]
# ## 1. Setup and Data Loading
#
# Let's start by importing our libraries and loading the Iris dataset, which contains measurements of 150 iris flowers from 3 different species.

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("Dataset Shape:", X.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nClass Distribution:")
print(df['species'].value_counts())

# %% [markdown]
# ## 2. Simple Train-Test Split
#
# ### The Concept
#
# The simplest validation approach is to split your data into two parts:
# - **Training Set** (typically 70-80%): Used to train the model
# - **Test Set** (typically 20-30%): Used to evaluate the model
#
# **Analogy:** It's like studying from a textbook (training) and then taking a final exam with different questions (testing).
#
# **Advantages:**
# - Simple and fast
# - Good for large datasets
#
# **Disadvantages:**
# - High variance in performance estimates (depends on which samples end up in test set)
# - Wastes data (test set isn't used for training)
# - Can be problematic with small datasets

# %%
# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Train a simple logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate on both sets
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(train_accuracy - test_accuracy):.4f}")

# %%
# Visualize the train-test split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data distribution
train_classes = np.bincount(y_train)
test_classes = np.bincount(y_test)

x_pos = np.arange(len(iris.target_names))
width = 0.35

axes[0].bar(x_pos - width/2, train_classes, width, label='Training Set', alpha=0.8)
axes[0].bar(x_pos + width/2, test_classes, width, label='Test Set', alpha=0.8)
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Number of Samples', fontsize=12)
axes[0].set_title('Class Distribution: Train vs Test', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(iris.target_names)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Accuracy comparison
accuracies = [train_accuracy, test_accuracy]
labels = ['Training', 'Test']
colors = ['#2ecc71', '#e74c3c']

bars = axes[1].bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Performance', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. K-Fold Cross-Validation
#
# ### The Concept
#
# Instead of a single train-test split, K-Fold Cross-Validation divides the data into **K equal-sized folds** and performs K training iterations:
#
# 1. Split data into K folds (e.g., K=5)
# 2. For each iteration i (1 to K):
#    - Use fold i as the test set
#    - Use remaining K-1 folds as the training set
#    - Train and evaluate the model
# 3. Average the K performance scores
#
# **Analogy:** Instead of one final exam, you take K different exams, each covering different material. Your final grade is the average.
#
# **Advantages:**
# - More reliable performance estimate (lower variance)
# - Every sample is used for both training and testing
# - Better for small datasets
#
# **Disadvantages:**
# - Computationally expensive (K times slower)
# - Still can have imbalanced folds in classification problems

# %%
# Function to perform K-Fold Cross-Validation
def evaluate_kfold(X, y, k_values=[3, 5, 10]):
    results = {}
    
    for k in k_values:
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=200)
        
        # Get scores for each fold
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        
        results[k] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"\n{k}-Fold Cross-Validation:")
        print(f"  Scores per fold: {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return results

# Evaluate with different K values
kfold_results = evaluate_kfold(X, y, k_values=[3, 5, 10])

# %%
# Visualize K-Fold results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Performance across folds for K=5
k = 5
fold_scores = kfold_results[k]['scores']
fold_numbers = np.arange(1, k + 1)

axes[0].plot(fold_numbers, fold_scores, marker='o', linewidth=2, markersize=10, label='Fold Accuracy')
axes[0].axhline(y=kfold_results[k]['mean'], color='r', linestyle='--', linewidth=2, label=f'Mean: {kfold_results[k]["mean"]:.4f}')
axes[0].fill_between(fold_numbers, 
                      kfold_results[k]['mean'] - kfold_results[k]['std'],
                      kfold_results[k]['mean'] + kfold_results[k]['std'],
                      alpha=0.2, color='red', label='±1 Std Dev')
axes[0].set_xlabel('Fold Number', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('5-Fold Cross-Validation: Performance per Fold', fontsize=14, fontweight='bold')
axes[0].set_xticks(fold_numbers)
axes[0].set_ylim([0.85, 1.0])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Comparison of different K values
k_values = list(kfold_results.keys())
means = [kfold_results[k]['mean'] for k in k_values]
stds = [kfold_results[k]['std'] for k in k_values]

axes[1].bar(range(len(k_values)), means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_xlabel('K Value', fontsize=12)
axes[1].set_ylabel('Mean Accuracy', fontsize=12)
axes[1].set_title('Impact of K on Cross-Validation Performance', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(k_values)))
axes[1].set_xticklabels([f'K={k}' for k in k_values])
axes[1].set_ylim([0.9, 1.0])
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    axes[1].text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variance Reduction with K-Fold
#
# Notice how K-Fold gives us multiple performance measurements instead of just one. This helps us:
#
# 1. **Understand stability**: If scores vary wildly across folds, our model might be sensitive to the training data
# 2. **Get confidence intervals**: The standard deviation tells us how reliable our estimate is
# 3. **Detect overfitting**: Large differences between training and validation scores indicate overfitting
#
# **Choosing K:**
# - **Smaller K (3-5)**: Faster, but higher variance in estimates
# - **Larger K (10)**: More reliable estimates, but computationally expensive
# - **K = N (Leave-One-Out)**: Maximum data usage, but very slow and high variance

# %% [markdown]
# ## 4. Stratified K-Fold Cross-Validation
#
# ### Why Stratification Matters
#
# Regular K-Fold has a potential problem: **folds might not have balanced class distributions**. This is especially critical when:
# - Dataset is small
# - Classes are imbalanced
# - Some classes are rare
#
# **Stratified K-Fold** ensures each fold maintains the same class proportions as the original dataset.
#
# **Example:** If your dataset has 60% class A and 40% class B, each fold will also have approximately 60% class A and 40% class B.
#
# **When to use:**
# - Classification problems (almost always)
# - Imbalanced datasets (absolutely necessary)
# - Small datasets (highly recommended)

# %%
# Compare regular K-Fold vs Stratified K-Fold
k = 5

# Regular K-Fold
kfold = KFold(n_splits=k, shuffle=True, random_state=42)
regular_scores = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=kfold)

# Stratified K-Fold
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
stratified_scores = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=stratified_kfold)

print("Regular K-Fold:")
print(f"  Scores: {[f'{s:.4f}' for s in regular_scores]}")
print(f"  Mean: {regular_scores.mean():.4f} (+/- {regular_scores.std() * 2:.4f})")

print("\nStratified K-Fold:")
print(f"  Scores: {[f'{s:.4f}' for s in stratified_scores]}")
print(f"  Mean: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")

print(f"\nVariance Reduction: {(regular_scores.std() - stratified_scores.std()) / regular_scores.std() * 100:.2f}%")

# %%
# Analyze class distribution in each fold
def analyze_fold_distributions(X, y, cv_splitter, cv_name):
    fold_distributions = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        test_labels = y[test_idx]
        class_counts = np.bincount(test_labels, minlength=len(iris.target_names))
        class_percentages = class_counts / len(test_labels) * 100
        fold_distributions.append(class_percentages)
    
    return np.array(fold_distributions)

# Get distributions
regular_dist = analyze_fold_distributions(X, y, kfold, "Regular")
stratified_dist = analyze_fold_distributions(X, y, stratified_kfold, "Stratified")

# Expected distribution (from full dataset)
expected_dist = np.bincount(y) / len(y) * 100

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

x = np.arange(k)
width = 0.25

for idx, (dist, title) in enumerate([(regular_dist, 'Regular K-Fold'), 
                                       (stratified_dist, 'Stratified K-Fold')]):
    for i, class_name in enumerate(iris.target_names):
        axes[idx].bar(x + i * width, dist[:, i], width, label=class_name, alpha=0.8)
        axes[idx].axhline(y=expected_dist[i], color=f'C{i}', linestyle='--', 
                         linewidth=2, alpha=0.5)
    
    axes[idx].set_xlabel('Fold Number', fontsize=12)
    axes[idx].set_ylabel('Class Percentage (%)', fontsize=12)
    axes[idx].set_title(f'{title}: Class Distribution per Fold', fontsize=14, fontweight='bold')
    axes[idx].set_xticks(x + width)
    axes[idx].set_xticklabels([f'Fold {i+1}' for i in range(k)])
    axes[idx].legend(title='Species')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nNotice how Stratified K-Fold maintains class balance across all folds!")
print("The dashed lines represent the expected distribution from the full dataset.")

# %% [markdown]
# ## 5. Train-Validation-Test Split
#
# ### The Three-Way Split
#
# When we need to tune hyperparameters, a simple train-test split isn't enough. We need three sets:
#
# 1. **Training Set (60%)**: Used to train the model
# 2. **Validation Set (20%)**: Used to tune hyperparameters and make model selection decisions
# 3. **Test Set (20%)**: Used ONLY for final evaluation (touched once at the very end)
#
# **Why three sets?**
# - If we tune hyperparameters using the test set, we're indirectly "training" on it
# - This leads to overly optimistic performance estimates
# - The test set must remain completely unseen until final evaluation
#
# **Workflow:**
# 1. Split data into train and temp (temp = validation + test)
# 2. Split temp into validation and test
# 3. Train multiple models with different hyperparameters on training set
# 4. Evaluate each on validation set
# 5. Select best hyperparameters
# 6. Train final model on train + validation
# 7. Evaluate once on test set

# %%
# Create three-way split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Dataset Split:")
print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Total: {len(X)} samples")

# %%
# Demonstrate hyperparameter tuning workflow
print("Hyperparameter Tuning: Decision Tree Max Depth\n")

# Try different max_depth values
depths = [2, 3, 4, 5, 6, 8, 10]
train_scores = []
val_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    
    print(f"Depth={depth:2d}: Train={train_acc:.4f}, Validation={val_acc:.4f}, Gap={train_acc-val_acc:.4f}")

# Select best depth based on validation performance
best_depth = depths[np.argmax(val_scores)]
print(f"\nBest max_depth based on validation set: {best_depth}")

# Train final model on train + validation
X_train_final = np.vstack([X_train, X_val])
y_train_final = np.hstack([y_train, y_val])

final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train_final, y_train_final)

# Evaluate on test set (only once!)
test_accuracy = final_model.score(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print("\n⚠️  Important: We only touch the test set once, at the very end!")

# %%
# Visualize the hyperparameter tuning process
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Training vs Validation curves
axes[0].plot(depths, train_scores, marker='o', linewidth=2, markersize=8, label='Training', color='#2ecc71')
axes[0].plot(depths, val_scores, marker='s', linewidth=2, markersize=8, label='Validation', color='#e74c3c')
axes[0].axvline(x=best_depth, color='gold', linestyle='--', linewidth=2, label=f'Best Depth={best_depth}')
axes[0].set_xlabel('Max Depth', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Hyperparameter Tuning: Train vs Validation Performance', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.85, 1.05])

# Plot 2: Overfitting gap
gaps = np.array(train_scores) - np.array(val_scores)
colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in gaps]

bars = axes[1].bar(range(len(depths)), gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_xlabel('Max Depth', fontsize=12)
axes[1].set_ylabel('Train - Validation Gap', fontsize=12)
axes[1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(depths)))
axes[1].set_xticklabels(depths)
axes[1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable Gap')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    axes[1].text(bar.get_x() + bar.get_width()/2., gap + 0.005, 
                f'{gap:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Comparison and Best Practices
#
# ### When to Use Each Method
#
# | Method | Best For | Avoid When |
# |--------|----------|------------|
# | **Train-Test Split** | Large datasets, quick experiments, baseline models | Small datasets, need robust estimates |
# | **K-Fold CV** | Model comparison, performance estimation, medium datasets | Very large datasets (slow), time-series data |
# | **Stratified K-Fold** | Classification (almost always!), imbalanced classes | Regression problems |
# | **Train-Val-Test** | Hyperparameter tuning, model selection, final evaluation | Simple experiments without tuning |
#
# ### Key Takeaways
#
# 1. **Always use stratification** for classification problems
# 2. **Never tune on test data** - it must remain unseen
# 3. **K-Fold reduces variance** but increases computation time
# 4. **Larger K ≠ always better** - balance bias-variance and speed
# 5. **For production**: Use K-Fold for model selection, then retrain on all data
#
# ### The Golden Rule
#
# **Test data is sacred** - touch it only once, at the very end, after all decisions are made!

# %%
# Final comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Computation time comparison (approximate)
methods = ['Train-Test', '5-Fold CV', 'Stratified\n5-Fold', 'Train-Val-Test']
relative_times = [1, 5, 5, 1]
colors_time = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']

axes[0, 0].bar(methods, relative_times, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Relative Computation Time', fontsize=12)
axes[0, 0].set_title('Computational Cost Comparison', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, time in enumerate(relative_times):
    axes[0, 0].text(i, time + 0.2, f'{time}x', ha='center', fontsize=11, fontweight='bold')

# Plot 2: Data usage efficiency
data_usage = [80, 100, 100, 60]  # Percentage of data used for training
axes[0, 1].bar(methods, data_usage, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('Data Used for Training (%)', fontsize=12)
axes[0, 1].set_title('Training Data Utilization', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim([0, 110])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, usage in enumerate(data_usage):
    axes[0, 1].text(i, usage + 2, f'{usage}%', ha='center', fontsize=11, fontweight='bold')

# Plot 3: Estimate reliability (inverse of variance)
reliability = [3, 9, 10, 5]  # Subjective score
axes[1, 0].bar(methods, reliability, color=colors_time, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Reliability Score', fontsize=12)
axes[1, 0].set_title('Performance Estimate Reliability', fontsize=14, fontweight='bold')
axes[1, 0].set_ylim([0, 11])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, rel in enumerate(reliability):
    axes[1, 0].text(i, rel + 0.3, f'{rel}/10', ha='center', fontsize=11, fontweight='bold')

# Plot 4: Use case suitability matrix
use_cases = ['Small\nDataset', 'Large\nDataset', 'Imbalanced\nClasses', 'Hyperparameter\nTuning']
suitability = np.array([
    [2, 4, 3, 2],  # Train-Test
    [4, 3, 3, 3],  # K-Fold
    [5, 4, 5, 4],  # Stratified K-Fold
    [3, 4, 4, 5],  # Train-Val-Test
])

im = axes[1, 1].imshow(suitability.T, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
axes[1, 1].set_xticks(range(len(methods)))
axes[1, 1].set_yticks(range(len(use_cases)))
axes[1, 1].set_xticklabels(methods, fontsize=10)
axes[1, 1].set_yticklabels(use_cases, fontsize=10)
axes[1, 1].set_title('Suitability Matrix (1=Poor, 5=Excellent)', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(methods)):
    for j in range(len(use_cases)):
        text = axes[1, 1].text(i, j, suitability[i, j], ha='center', va='center', 
                              color='white' if suitability[i, j] > 3 else 'black',
                              fontsize=12, fontweight='bold')

plt.colorbar(im, ax=axes[1, 1], label='Suitability Score')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# Congratulations! You've learned about the four main validation techniques in machine learning:
#
# 1. **Train-Test Split**: Simple and fast, good for quick experiments
# 2. **K-Fold Cross-Validation**: More reliable estimates, better for model comparison
# 3. **Stratified K-Fold**: Essential for classification, especially with imbalanced data
# 4. **Train-Validation-Test**: Required for hyperparameter tuning and final evaluation
#
# ### Next Steps
#
# - Practice with your own datasets
# - Try different K values and observe the variance
# - Always use stratification for classification
# - Remember: test data is sacred!
#
# Happy learning! 🎉