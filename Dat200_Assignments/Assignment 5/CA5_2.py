#!/usr/bin/env python
# coding: utf-8

# ## Importing

# In[230]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, RANSACRegressor, BayesianRidge, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import  mean_absolute_error
from xgboost import XGBClassifier



# To control warning messages. I should have done this last time to make your life easier!
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', "ConvergenceWarning")


# # Reading dataset

# In[231]:


# Load data
df = pd.read_csv("train.csv")


# ## Visualisation and data exploring

# In[232]:


# for comparing different features to each other from the dataset, with hue as 'Scoville Heat Units (SHU)'

sns.pairplot(df, hue='Scoville Heat Units (SHU)')
plt.legend
plt.show()


# In[233]:


#Checking the distorbution of each feature

for column in df.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True)  # kde (Kernel Density Estimate) adds a density curve
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[234]:


df.describe()


# In[235]:


#finding missing values in each feature
nan_per_column = df.isna().sum()
#finding totaol missing values in dataset
total_nan = df.isna().sum().sum()

# To display the number of NaNs per column
print(nan_per_column)
# To display the total number of NaNs in the DataFrame
print(f"Total number of NaN values in the DataFrame: {total_nan}")


# # !!!Comments on visualisation of data!!!

# There was a lot of missing values on the feature "Average Temperature During Storage (celcius)", so i'm removing it. Im replacing the rest of the missing values in the dataset with the median in the preprocesser
# 
# It looks like there wasn't much outliers in the dataset. Maybe som big values in 'Average Temperature During Storage (celcius)', so i set a threshold there for max values. It worked well on the models i used at least.
# 
# There is some categorical data that need transformation. 
# 
# I tried to put together features like Weight and lenght into one feature, but it didn't make much difference, so i didn't stick with it.

# # Data cleaning and preprocessing

# In[236]:


#Dropping the dataset with a lot if missing values
# I'm using SimpleImputer in the preprocesser that will replace missing values.
df.drop('Average Temperature During Storage (celcius)', axis=1, inplace=True)
# Setting a threshold for 'Vitamin C Content (mg)' at 400. 
df.drop(df[df['Vitamin C Content (mg)'] > 400].index, inplace=True)


# In[237]:


# Separate features and target
X = df.drop('Scoville Heat Units (SHU)', axis=1)
y = df['Scoville Heat Units (SHU)']

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

#making the preprocesser, that handles both numerical and categorical data
preprocessor = ColumnTransformer([
    ('num', Pipeline([ #for numerical data
        ('imputer', SimpleImputer(strategy='median')), #replacing missing values with median of the feature
    ]), numerical_cols),
    ('cat', Pipeline([ #for categorical data
        ('encoder', OneHotEncoder(handle_unknown='ignore')) #For transforming categorical data into numeric values
    ]), categorical_cols) 
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#selecting y/y_train/y_test_binary as values bigger than 1 for the ensemble classifier
y_train_binary = (y_train > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)
y_binary= (y > 0).astype(int)


# ## Model selecting 

# Here i select the models that works best for the dataset! I want to make a regression pipeline (A) and two seqential pipelines, one ensemble classifier and one regression classifier (C). I only included the model selector for (C), because it's literally the same procedure as (A), with only difference that i choose a combination of models with best MAE-score

# In[248]:


# Base estimators for stacking and voting
base_estimators = [
    ('lr', Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(random_state=42, max_iter=1000)) ])),
    ('svc', SVC(probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Create a decision tree classifier with max_depth=2
base_estimator = DecisionTreeClassifier(max_depth=2)


# Lists of ensemble models to try
classifiers = [
    XGBClassifier(),
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    #ExtraTreesClassifier(random_state=42), # Not sure if allowed
    BaggingClassifier(random_state=42),
]

#List of regressors to try
regressors = [
    RANSACRegressor(random_state=42),
    LinearRegression(),
    Ridge(random_state=42),
    ElasticNet(random_state=42),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    BayesianRidge(),
    SVR(),
    Pipeline([
        ('polynomial_features', PolynomialFeatures()),
        ('linear_regression', LinearRegression())
    ]),
    Pipeline([
        ('pca', PCA()),  # Adjust n_components as needed
        ('linear_regression', LinearRegression())
    ])
]


# In[249]:


# For scoring best score and info
best_score = np.inf
best_combo = None

#Kfold cross validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

#iterating over diferent classifiers in list
for clf in classifiers:
    # Create a unique name for the classifier pipeline for easy identification
    pipeline_clf_name = f'{clf.__class__.__name__}_pipeline'
    #define pipeline for ensemble classifier
    pipeline_clf = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', clf)
    ])

    #iterating over different classifiers in list
    for reg in regressors:
         # Create a unique name for the classifier pipeline for easy identification
        pipeline_reg_name = f'{reg.__class__.__name__}_pipeline'
        # Create pipeline for the regressor with a name
        pipeline_reg = Pipeline([
            ('preprocessing', preprocessor),
            ('scaling', StandardScaler()),
            ('regressor', reg)
        ])
        
        # Perform cross-validation

         # List to store the Mean Absolute Error for each fold
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Convert the target variable to binary (1 for spicy, 0 for not spicy)
            y_train_binary = (y_train > 0).astype(int)

            # Fit the classifier pipeline on the training data
            pipeline_clf.fit(X_train, y_train_binary)
            # Identify indices of spicy samples to train the regressor only on these
            spicy_peppers_indices = (pipeline_clf.predict(X_train) == 1)

             # Fit the regressor pipeline only on spicy samples
            pipeline_reg.fit(X_train[spicy_peppers_indices], y_train[spicy_peppers_indices])

             # Get predictions for spicy/not spicy from the classifier
            binary_predictions = pipeline_clf.predict(X_test)

             # Get regression predictions for the test set
            regression_predictions = pipeline_reg.predict(X_test)

             # Combine predictions: Only apply regression predictions where the classifier predicts spicy
            combined_predictions = regression_predictions * binary_predictions
            
            # Calculate Mean Absolute Error for the combined predictions
            mae = mean_absolute_error(y_test, combined_predictions)
            scores.append(mae)

         # Calculate the average MAE across all folds
        average_mae = np.mean(scores)
        # Update the best score and store the best classifier/regressor combination if this combo is better
        if average_mae < best_score:
            best_score = average_mae
            best_combo = (clf, reg)
            best_combo_names = (pipeline_clf_name, pipeline_reg_name)

print(f'Best combination: {best_combo_names[0]} + {best_combo_names[1]} with average MAE {best_score}')


# ## Pipeline (A)

# In[250]:


# Making a pipeline with LinearRegression()
regression_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing
    ('scaling', StandardScaler()),  # Scaling features
    ('regression', LinearRegression())  # Linear regression model
])

# Fit the model
regression_pipeline.fit(X_train, y_train)

# Predict using the model
regression_predictions = regression_pipeline.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, regression_predictions)

print(f"Mean Absolute Error (MAE) on Test Data: {mae}")


# ## Pipelines for (C) (Please read under)

# So i used ExtraTreesClassifier and Ridge classifiers for (C). Then i realized that we havent learning about ExtraTreesClassifier in this course, so if it isn't allowed then my best score on kaggle shouldn't count. Ive made pipelines with Gradientboosting and ElasticNet if ExtraTreeClassifier isn't allowed.

# ## My best combination of models 

# In[251]:


# Define the pipeline for binary classifier 
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', ExtraTreesClassifier(random_state=42))])

# Define the hyperparameters grid to be tuned
params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
}

# Setup the GridSearchCV object
binary_classifier_pipeline = GridSearchCV(estimator=pipeline, 
                           param_grid=params, 
                           cv=5, 
                           scoring='neg_mean_absolute_error', 
                           verbose=1, 
                           n_jobs=-1)


# In[252]:


# # Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('scaling', StandardScaler()),
                           ('regressor', Ridge(random_state=42))])

# Define the hyperparameters grid to be tuned
params = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
    # Add other parameters you want to tune
}

# Setup the GridSearchCV object
regression_pipeline = GridSearchCV(estimator=pipeline, 
                           param_grid=params, 
                           cv=5, 
                           scoring='neg_mean_absolute_error', 
                           verbose=1, 
                           n_jobs=-1)


# ## Alternative Models

# In[261]:


# This is the binary classifier pipeline im using if ExtraTreeClassifier isn't allowed.


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define the hyperparameters grid to be tuned for the GradientBoostingClassifier
params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],  # Learning rate to tune
    'classifier__max_depth': [3, 5, 7],  # Depths to tune
    # You can add other parameters here to tune
}

# Setup the GridSearchCV object for the binary classifier
binary_classifier_pipeline = GridSearchCV(estimator=pipeline,
                                          param_grid=params,
                                          cv=5,
                                          scoring='neg_mean_absolute_error',
                                          verbose=1,
                                          n_jobs=-1)


# In[264]:


#This is the regressor pipeline im using if ExtraTreeClassifier isn't allowed

# Define the pipeline with an Elastic Net regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaling', StandardScaler()),
    ('regressor', ElasticNet(random_state=42))
])

# Define the hyperparameters grid to be tuned for the Elastic Net
params = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'regressor__l1_ratio': [0.1, 0.5, 0.9],       # Mix ratio of L1 and L2 regularization
}

# Setup the GridSearchCV object for the regression pipeline
regression_pipeline = GridSearchCV(estimator=pipeline,
                                   param_grid=params,
                                   cv=5,
                                   scoring='neg_mean_absolute_error',
                                   verbose=1,
                                   n_jobs=-1)


# ## Training and evaluation of models

# In[265]:


# Fitting binary classifier with trainig sets.
binary_classifier_pipeline.fit(X_train, y_train_binary)


# In[266]:


# To estimate the scoville score (SHU) of those samples that the binary classifier identifies as spicy peppers.
spicy_peppers_indices = (binary_classifier_pipeline.predict(X_train) == 1)

# Training the model
regression_pipeline.fit(X_train[spicy_peppers_indices], y_train[spicy_peppers_indices])

# Combine predictions of both models into a single prediction vector
binary_predictions = binary_classifier_pipeline.predict(X_test)
regression_predictions = regression_pipeline.predict(X_test)

# Combine predictions for spicy peppers (SHU > 0)
combined_predictions = regression_predictions * binary_predictions

# Evaluate the combined model
mae_combined = mean_absolute_error(y_test, combined_predictions)
print("Mean Absolute Error (Combined Model):", mae_combined)


# In[267]:


# Fit all the data for the binary classifier
binary_classifier_pipeline.fit(X, y_binary)


# In[268]:


spicy_peppers_indices = (binary_classifier_pipeline.predict(X) == 1)
#Fit all the data for the regresssion classifier
regression_pipeline.fit(X[spicy_peppers_indices], y[spicy_peppers_indices])


# ## Kaggle Submission

# In[269]:


#Load test set
X_test = pd.read_csv("test.csv")

#Use the trained pipelines to make predictions on the test set
binary_predictions = binary_classifier_pipeline.predict(X_test)
regression_predictions = regression_pipeline.predict(X_test)

#Combine predictions of both models into a single prediction vector
combined_predictions = regression_predictions * binary_predictions

# Create a DataFrame with the index and predicted SHU values
results_df = pd.DataFrame({'index': range(len(X_test)), 'Scoville Heat Units (SHU)': combined_predictions})

# Save the DataFrame to a CSV file with the specified format
results_df.to_csv('predicted_shu.csv', index=False)

# Display the first few rows of the generated file
print(results_df.head())

