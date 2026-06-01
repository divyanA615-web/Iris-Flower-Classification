# ===============================================
# Task 1: IRIS DATASET - EDA + CLASSIFICATION
# ===============================================

import pandas as pd
import numpy as np # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay)  # type: ignore

# -- 1.load the dataset --
Iris = pd.read_csv(r"D:\data science related\iris_flower_classification\Iris-Flower-Classification\Iris.csv")
print("=== IRIS DATASET LOADED===")
print(f"shape: {Iris.shape}")
print(f"columns: {Iris.columns.to_list()}")
print(f"\nFirst 5 rows:\n{Iris.head()}")

#-- 2. clean the dataset --
print(f"\n--- Missing Values ---\n{Iris.isnull().sum()}")
print(f"\n--- Duplicate Rows ---\n{Iris.duplicated().sum()} ---")
Iris = Iris.drop_duplicates()
Iris = Iris.drop(columns=['Id']) #Id is not a feature

#-- 3. Statistics --
print(f"\n--- Descriptive Statistics ---\n{Iris.describe().round(2)}")
print(f"\n--- Class Distribution ---\n{Iris['Species'].value_counts()}")

#-- 4.Chart 1 - Feature Distributions (Histograms) --

fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # type: ignore
fig.suptitle('🌸 Iris Feature Distributions', fontsize=16, fontweight='bold') # type: ignore
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
colors = ["#3980F1", '#DD8452', '#55A868', '#C44E52']
for ax, feat, color in zip(axes.flatten(), features, colors):
    Iris[feat].hist(ax=ax, bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('cm')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Iris_Feature_Distributions.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Feature_Distributions.png")

# -- 5. Chart 2 - Pairplot by Species --
palette = {'Iris-setosa': '#E74C3C',
           'Iris-versicolor': '#3498DB',
           'Iris-virginica': '#2ECC71'}
pair = sns.pairplot(Iris, hue='Species', palette=palette, diag_kind='kde', plot_kws={'alpha':0.7, 's':60}) # type: ignore
pair.figure.suptitle('🌸 Iris Pairplot by Species', y=1.02, fontsize=16, fontweight='bold') # type: ignore
plt.savefig('Iris_Pairplot.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Pairplot.png")

# -- 6. Chart 3 - Correlation Heatmap --
fig, ax = plt.subplots(figsize=(7, 5)) # type: ignore
corr = Iris.drop("Species", axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn',mask=mask, linewidths=0.5, ax=ax,annot_kws={"size": 12, "weight": 'bold'}) # type: ignore
ax.set_title('🌸 Iris Feature Correlation Heatmap', fontsize=13, fontweight='bold') # type: ignore
plt.tight_layout() # type: ignore
plt.savefig('Iris_Correlation_Heatmap.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Correlation_Heatmap.png")

# -- 7. Chart 4 - Box Plot of Features by Species --
fig, axes = plt.subplots(2, 2, figsize=(13, 9)) # type: ignore
fig.suptitle('🌸 Feature Distributions by Species(Boxplots)',fontsize=14, fontweight='bold') # type: ignore
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=Iris, x='Species', y=feat,ax=ax, palette=palette,linewidth=1.5) # type: ignore
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15) # type: ignore
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout() # type: ignore
plt.savefig('Iris_Boxplots.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Boxplots.png")

# -- 8. Machine Learning - Random Forest Classifier --
X = Iris[features]
le = LabelEncoder()
y = le.fit_transform(Iris['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # type: ignore
model = RandomForestClassifier(n_estimators=100, random_state=42) # type: ignore
model.fit(X_train, y_train) # type: ignore
y_pred = model.predict(X_test) # type: ignore

acc = accuracy_score(y_test, y_pred) # type: ignore
print(f"\n{'='*45}")
print(f"🎯 Random Forest Accuracy: {acc*100:.2f}%")
print(f"\n{'='*45}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_)) # type: ignore

# -- 9.Chart 5 -Confusion Matrix --
fig, ax = plt.subplots(figsize=(7, 5)) # type: ignore
cm = confusion_matrix(y_test, y_pred) # type: ignore
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_) # type: ignore
disp.plot(ax=ax, colorbar=True, cmap='Blues') # type: ignore
ax.set_title(f'🎯 Confusion Matrix(Accuracy: {acc*100:.1f}%)', fontsize=12,fontweight='bold') # type: ignore
plt.tight_layout() # type: ignore
plt.savefig('Iris_Confusion_Matrix.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Confusion_Matrix.png")

# -- 10. Chart 6 - Feature Importance --
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True) # type: ignore
fig, ax = plt.subplots(figsize=(8, 4)) # type: ignore
importances.plot(kind='barh', ax=ax, color=['#C44E52', '#55A868', '#DD8452', '#4C72B0']) # type: ignore
ax.set_title('🎯 Feature Importance(Random Forest)', fontsize=13, fontweight='bold') # type: ignore
ax.set_xlabel('Importance Score') # type: ignore
for i, v in enumerate(importances):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontweight='bold') # type: ignore
plt.tight_layout() # type: ignore
plt.savefig('Iris_Feature_Importance.png', dpi=150, bbox_inches='tight') # type: ignore
plt.close()
print("✅ Saved: Iris_Feature_Importance.png")

print("\n=== Task 1 Completed: EDA + Classification on Iris Dataset ===")