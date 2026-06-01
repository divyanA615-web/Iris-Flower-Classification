# ===============================================
# Task 1: IRIS DATASET - EDA + CLASSIFICATION
# ===============================================

import pandas as pd
import numpy as np # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import (
accuracy_score, # type: ignore
classification_report, # type: ignore
confusion_matrix, # pyright: ignore[reportUnusedImport] # type: ignore # type: ignore # type: ignore
ConfusionMatrixDisplay) # type: ignore

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

