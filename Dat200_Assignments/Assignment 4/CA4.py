#!/usr/bin/env python
# coding: utf-8

# Imports

# In[119]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import expon, loguniform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV


# ## Reading data

# In[120]:


# Load data
df = pd.read_csv("train.csv", index_col=0)
df = df.drop(['index'], axis=1)
df


# ### Data exploration and visualisation

# In[121]:


# for comparing different features to each other from the dataset, with hue as Edible
#I just made comments out of this so you dont have to read all the output

# sns.pairplot(df, hue='Diagnosis')
# plt.legend
# plt.show()


# In[122]:


# For looking at the distribution of the data of each feature in the dataset 
# I just made comments out of this so you dont have to read all the output

# for column in df.columns:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[column], kde=True)  # kde (Kernel Density Estimate) adds a density curve
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()


# In[123]:


df.describe()


# In[124]:


nan_per_column = df.isna().sum()
total_nan = df.isna().sum().sum()

# To display the number of NaNs per column
print(nan_per_column)

# To display the total number of NaNs in the DataFrame
print(f"Total number of NaN values in the DataFrame: {total_nan}")


# ## Comments on the visualisation

# I found no missing values. There was features that had some outliers, but maybe some of them have their purpose, so i haven't removed them. There was some negative values in the dataset that didn't make sense, so i'm removing them

# ### Data cleaning

# In[125]:


# Replace negative values with zero for specified columns
for column in ['AFP (ng/mL)', 'ALT (U/L)', 'AST (U/L)']:
    df[column] = df[column].apply(lambda x: max(x, 0))


# ## Preprocessing

# In[126]:


# Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing steps, i make it ignore unkown categories in the OneHotEncoder
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),  # Apply standard scaling to numerical columns
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Apply one-hot encoding to categorical columns
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Modelling 

# In[128]:


# SVM Pipeline, focusing on getting the highest f1-macro score 
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor), #adding the preprocesser 
    ('feature_selector', SFS(SVC(kernel='rbf', random_state=42),  #using rbf-kernel 
                             k_features=(5, 20), #the different combos of features i want to try
                             forward=True, #forward SFS
                             floating=True, #allowing both inclusion and exclusion of features during the  process
                             scoring='f1_macro', # getting the highest f1-macro score
                             cv=5)), #5-fold cross validation
    ('classifier', RandomizedSearchCV(SVC(kernel='rbf', random_state=42),
                                      {'C': expon(scale=1), #Finding the best value for C
                                       'gamma': ['scale', 'auto'] + list(expon(scale=0.1).rvs(size=20))}, #finding the best value for gamma
                                      n_iter=200, #how many iteartions 
                                      cv=10, #10-Fold Cross-Validation, because i felt like it
                                      scoring='f1_macro', # Focusing on getting the best f1_macro score
                                      random_state=42, 
                                      n_jobs=-1)) #using all processors avaliable when running 
                                      ])

# Fit the pipeline
svm_pipeline.fit(X_train, y_train) #fitting the training, so i can evaluate

# Evaluate the model
y_pred_svm = svm_pipeline.predict(X_test)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
print("F1-Score (Macro):", f1_svm) # getting the F1 score from the evaluation

# getting the best paramters from the evalutation
best_params_svm = svm_pipeline.named_steps['classifier'].best_params_
print("Best Parameters:", best_params_svm)


# After using RandomSearch, i tried grid sarch on a smaller range around the best paramters i've found earlier. 

# In[129]:


param_grid = {
    'C': np.linspace(best_params_svm['C'] * 0.5, best_params_svm['C'] * 2, 10),  # Creating 10 points between half and twice the best 'C'
    'gamma': np.linspace(best_params_svm['gamma'] * 0.5, best_params_svm['gamma'] * 2, 10)  # Creating 10 points between half and twice the best 'gamma'
}

# Since we are doing a grid search, we do not need the RandomizedSearchCV part anymore.
# We directly plug in the GridSearchCV in place of the RandomizedSearchCV in the pipeline
svm_pipeline.named_steps['classifier'] = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=10,  # 10-Fold Cross-Validation
    scoring='f1_macro',
    n_jobs=-1
)

# Fit the pipeline with the new GridSearchCV
svm_pipeline.fit(X_train, y_train)

# Evaluate the model with the updated classifier
y_pred_svm = svm_pipeline.predict(X_test)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
print("F1-Score (Macro) with Grid Search:", f1_svm)

# Get the best parameters from the grid search
best_params_svm_grid = svm_pipeline.named_steps['classifier'].best_params_
print("Best Parameters from Grid Search:", best_params_svm_grid)


# In[ ]:


# Logistic Regression Pipeline, same procedure, just a different classifier 
logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SFS(LogisticRegression(random_state=42), 
                             k_features=(5, 20), 
                             forward=True, 
                             floating=False, 
                             scoring='f1_macro', 
                             cv=5)),
    ('classifier', RandomizedSearchCV(LogisticRegression(random_state=42),
                                      {'C': loguniform(1e-4, 1e2), #getting the best value for C
                                       'penalty': ['l1', 'l2']}, #finding out which regularization works best
                                      n_iter=50, 
                                      cv=5, 
                                      scoring='f1_macro',  
                                      random_state=42, 
                                      n_jobs=-1))
])

logreg_pipeline.fit(X_train, y_train)

y_pred_lr = logreg_pipeline.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
print("F1-Score (Macro):", f1_lr)

best_params_lr = logreg_pipeline.named_steps['classifier'].best_params_
print("Best Parameters:", best_params_lr)

#I made comments on the earlier pipeline


# ## Confusion matrix for the best pipeline

# SVM-pipeline performed best! Now i'm making a confusion matrix for the best pipeline.

# In[ ]:


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fit the pipeline
svm_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred_full = svm_pipeline.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_full)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


# Fit the svm-model with the whole dataset
svm_pipeline.fit(X, y)


# In[ ]:


#fit the logreg model with the whole dataset
logreg_pipeline.fit(X, y)


# ## Kaggle submission

# In[ ]:


# Evaluate and save results
X_test = pd.read_csv("test.csv").drop(['index'], axis=1)
svm_predictions = svm_pipeline.predict(X_test)
# logreg_predictions = logreg_pipeline.predict(X_test)

# Save predictions to CSV files
pd.DataFrame({'index': X_test.index, 'Diagnosis': svm_predictions}).to_csv('svm_predictions.csv', index=False)
# pd.DataFrame({'index': X_test.index, 'Diagnosis': logreg_predictions}).to_csv('logreg_predictions.csv', index=False)


# ## Seperate part of task

# In[ ]:


# Load data again, so the dataset is right every time i run this box
df = pd.read_csv("train.csv", index_col=0)
df = df.drop(['index'], axis=1)

# Replace negative values with zero for specified columns
for column in ['AFP (ng/mL)', 'ALT (U/L)', 'AST (U/L)']:
    df[column] = df[column].apply(lambda x: max(x, 0))

#Creating Condition binary 
df['ConditionBinary'] = df['Diagnosis'].apply(lambda x: 0 if x == 'Healthy' else 1) #Will be 0 if healty and 1 if sick 
df = df.drop(['Diagnosis'], axis=1)

# Separate features and target
X = df.drop('ConditionBinary', axis=1)
y = df['ConditionBinary']

# Initializes Stratified K-Fold cross-validator with 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up pipeline
pipe_lr = make_pipeline(preprocessor,
                        LogisticRegression(penalty='l2', random_state=42))

# Obtain predicted probabilities using cross-validation
y_probas = cross_val_predict(pipe_lr, X, y, cv=skf, method='predict_proba')

# Compute ROC AUC scores directly during cross-validation
auc_scores = cross_val_score(pipe_lr, X, y, cv=skf, scoring='roc_auc')

# Compute mean and standard deviation of the AUC scores
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_probas[:, 1])

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

