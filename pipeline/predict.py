import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# custom files
import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("../data/new_data.csv")
print('new data size', ds.shape)

# feature engineering
param_dict = pickle.load(open('param_dict.pickle', 'rb'))

# Missing data imputation
def impute_na(df, variable, value):
    return df[variable].fillna(value)

# Let's read a dict and impute median values
for column in columns.median_impute_columns:
    ds[column] = impute_na(ds, column, param_dict['median_impute_values'][column])

# Categorical encoding    
le = LabelEncoder()

for column in columns.cat_columns:
    if column != "satisfaction":
    	ds[column] = le.fit_transform(ds[column])

# Outlier Engineering
# Ініціалізація моделі LocalOutlierFactor
lof = LocalOutlierFactor(contamination=0.05)  # Встановіть contamination залежно від розміру викидів, які ви очікуєте
# Обчислення оцінки аномалій (викидів) для кожного прикладу в датасеті
outlier_scores = lof.fit_predict(ds[columns.numerical_columns])

# Відображення кількості викидів
outlier_count = len(ds[outlier_scores == -1])
print("Number of outliers:", outlier_count)

# Видалення рядків, що містять викиди
ds = ds[outlier_scores != -1]

# Відображення оновленого датасету без викидів
print(f"Dataset without outliers:{len(ds)}")

# Define features columns
X = ds[columns.X_columns]

# load the model and predict
rf = pickle.load(open('finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['satisfaction_pred'] = rf.predict(X)
ds.to_csv('prediction_results.csv', index=False)

print(ds.head(20))