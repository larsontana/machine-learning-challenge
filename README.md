# machine-learning-challenge

# Import libraries and dependencies.
import pandas as pd
# Clean up data
df = pd.read_csv("exoplanet_data.csv")
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()

df.describe()

# Select features
# Set target, features and feature_names.
target = df["koi_disposition"]
data = df.drop("koi_disposition", axis=1)
feature_names = data.columns
data.head()

# Create a train test split. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)
# This splits arrays into random subsets

# Pre Processing
from sklearn.preprocessing import MinMaxScaler
X_minmax = MinMaxScaler().fit(X_train)

X_train_minmax = X_minmax.transform(X_train)
X_test_minmax = X_minmax.transform(X_test)

# Train the model 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_minmax, y_train)

print(f"Training Data Score: {rf.score(X_train_minmax, y_train)}")
print(f"Testing Data Score: {rf.score(X_test_minmax, y_test)}")

sorted(zip(rf.feature_importances_, feature_names), reverse=True)

# Hyper paramater tuning
# Create the GridSearchCV model
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [250, 300, 350],
              'max_depth': [125, 150, 175]}
grid = GridSearchCV(rf, param_grid, verbose=3)

# Train the model with GridSearch
grid.fit(X_train_minmax, y_train)

print(grid.best_params_)
print(grid.best_score_)

# Training score:
grid.score(X_train_minmax, y_train)

# Testing score:
grid.score(X_test_minmax, y_test)

# Make prediction and save to variable for report.
predictions = grid.predict(X_test_minmax)

# Print Classification Report.
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# %matplotlib notebook
from yellowbrick.classifier import ClassificationReport
viz = ClassificationReport(RandomForestClassifier())
viz.fit(X_train_minmax, y_train)
viz.score(X_test_minmax, y_test)
viz.finalize()
viz.show(outpath="Images/randomforest_classificationreport.png")

from yellowbrick.model_selection import FeatureImportances
from yellowbrick.style import set_palette
from yellowbrick.features import RadViz
set_palette('yellowbrick')
viz = FeatureImportances(rf, size=(500, 500))
viz.fit(X_train_minmax, y_train)
viz.show(outpath="Images/featureimportance.png")

# Save the model
import joblib
filename = 'Models/zGrinacoff_randomForest.sav'
joblib.dump(rf, filename)
