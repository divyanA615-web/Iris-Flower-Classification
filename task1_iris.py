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