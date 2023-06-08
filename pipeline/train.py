import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import LocalOutlierFactor
# custom files

import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("../data/train_data.csv")

# feature engineering

# Missing data imputation

def impute_na(df, variable, value):
    return df[variable].fillna(value)

# Let's create a dict and impute median values
median_impute_values = dict()
for column in columns.median_impute_columns:
    median_impute_values[column] = ds[column].median()
    ds[column] = impute_na(ds, column, median_impute_values[column])
    
for col in ds.columns:
    if ds[col].isnull().values.any():
        print("Missing data in ", col, ds[col].isnull().sum())
    else:
        print("There aren't missing data")

# for one hot encoding with sklearn

le = LabelEncoder()

for column in columns.cat_columns:
    ds[column] = le.fit_transform(ds[column])
    
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

# save parameters 
param_dict = {'median_impute_values':median_impute_values}
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

# Building and train Random Forest Model
rf = RandomForestClassifier(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('test set metrics: ', metrics.classification_report(y_test, y_pred))

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))