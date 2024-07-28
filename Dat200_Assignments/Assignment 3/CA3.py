#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1022]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ### Reading data

# In[1023]:


df= pd.read_csv("train.csv",index_col=0 )


# ### Data exploration and visualisation

# In[1024]:


# # for comparing different features to each other from the dataset, with hue as Edible
# #I just made comments out of this so you dont have to read all the output

# sns.pairplot(df, hue='Edible')
# plt.show()


# In[1025]:


#For looking at the distribution of the data of each feature in the dataset 
#I just made comments out of this so you dont have to read all the output

# for column in df.columns:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[column], kde=True)  # kde (Kernel Density Estimate) adds a density curve
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()


# In[1026]:


#Looking for correlation between features in the dataset 
#I just made comments out of this so you dont have to read all the output

# corr_matrix = df.corr()
# # Apply formatting to round the values to two decimal places.
# # The .applymap() method applies a lambda function to each element of the correlation matrix.
# # The lambda function converts each value 'x' in the matrix to a formatted string with two decimal places.
# corr_matrix_rounded = corr_matrix.applymap(lambda x: f'{x:.2f}')

# # Plotting the heatmap, first setting the size of the heatmap
# plt.figure(figsize=(10, 8))
# # 'annot=corr_matrix_rounded' is used to display the rounded correlation values as annotations on the heatmap.
# # 'fmt=' specifies that the annotation texts (rounded values) are strings, not floats or integers.
# # 'cmap='coolwarm' sets the colormap to 'coolwarm' 
# sns.heatmap(corr_matrix, annot=corr_matrix_rounded, fmt='', cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()


# In[1027]:


# Looking at df to see how many rows and columns there is 
#I just made comments out of this so you dont have to read all the output
#df


# In[1028]:


df.describe()


# #### Comments on the data after visualization
# Some of the pH values didn't look right. There was some negative ph values, which doesn't make sense.
# It looked like there was some outliers on some of the features, which is also getting removed before training.

# ### Data cleaning

# In[1029]:


#replacing nan-values with median
# I do this instead of removing the rows, because it gave me better accuracy when training models
df_cleaned= df.fillna(df.median())

# Dropping rows with negative pH values
df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['pH'] < 0].index)

#Dropping rows with feature values that clearly looks like outliers.
# I did this manually by just looking at the plots, but i could've also removed data with z-score
df_cleaned= df_cleaned.drop(df_cleaned[df_cleaned["Acoustic Firmness Index"] > 50].index)
df_cleaned= df_cleaned.drop(df_cleaned[df_cleaned["Odor index (a.u.)"] > 85].index)

#I've could've also dropped these rows under, but the accuracy got worse when doing this
#df_cleaned= df_cleaned.drop(df_cleaned[df_cleaned["Luminescence Intensity (a.u.)"] > 0.017].index)

# For checking how many rows i've removed
#I just made comments out of this so you dont have to read all the output
#df_cleaned


# ### Data preprocessing and visualisation

# In[1030]:


# Plotting the distribution of pH values after data-cleaning
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["pH"], kde=True)
plt.title('Distribution of pH without outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# In[1031]:


# Plotting the distribution of Acoustic Firmess Index after data-cleaning
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["Acoustic Firmness Index"], kde=True)
plt.title('Distribution of Acoustic Firmess Index without outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# In[1032]:


# Plotting the distribution of Odor index (a.u.) after data-cleaning
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["Odor index (a.u.)"], kde=True)
plt.title('Distribution of Odor index (a.u.) without outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# In[1033]:


# Removing the target column from the DataFrame and uses the rest as features
X = df_cleaned.drop('Edible', axis=1)  
# Selcts Edible as target columns 
y = df_cleaned['Edible']

#Splitting up into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialise standard scaler and compute mean and stddev from training data
sc = StandardScaler()
sc.fit(X_train)


# Transform (standardise) both X_train and X_test with mean and stddev from
# training data, to avoid leakage 
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)


# ### Modelling

# In[1034]:


# No need to standardize the data for Random Forest 

# Trying ot RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Making predictions with training data
y_pred_train = rfc.predict(X_train)
#computing accuracy from training data
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'Accuracy on training data: {accuracy_train:.2f}')

# Making predictions with test data
y_pred = rfc.predict(X_test)
#computing accuracy from test data
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data: {accuracy:.2f}')


# Here i tried different classifiers, but i took it out so you have less code to look at!

# ### Final evaluation

# Got best accuracy from randomforest, so i use this classifier

# In[1041]:


# Getting feature importance
feature_importances = rfc.feature_importances_

# Creating a DataFrame for feature importances
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sorting the features by importance
features_df = features_df.sort_values(by='Importance', ascending=False)
print("Feature importances:\n", features_df)


# In the next box i try all different combinations of features. I use the feautre combination that has the highest accuracy 

# In[1042]:


# Selecting subsets of features starting from the most important
best_average_accuracy = 0
best_features = []


for i in range(1, len(features_df.Feature) + 1):
    #Selecting the combination to use for the model
    selected_features = features_df.Feature.head(i)
    X_selected = X[selected_features]

    #storing accuracy 
    accuracies = []
    
    # Iterate over a range of random states for train-test split
    for r in range(1,100, 10):
        # Splitting data with varying random states, to avoid overfitting
        #splitting with only the selected features
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=r)
        
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    
    # Calculate the average accuracy across all splits for this set of features
    average_accuracy = np.mean(accuracies)
    
    # Update the best feature-lsit if the current average accuracy is better
    if average_accuracy > best_average_accuracy:
        best_average_accuracy = average_accuracy
        best_features = selected_features.tolist()

print(f"Best Average Accuracy: {best_average_accuracy:.4f}")
print(f"Best Features: {best_features}")


# In the next box, i try to find the best paramters for the model. I started with looping over n_estimators, and selected the value with the best accuracy. Then i adjusted the code with best_n_estimators, and looped over different values of max_depth. I select the max_depth-value that gave me the highest accuraccy with the best_n_estimators. This goes on in the same pattern for the next parameter.

# I've tried using different criterions and random_states for the model, and earlier i've found that criterion= "gini" and randomstate= 42 has worked well for me..

# I could have tried to find more paramters to use in the model, but the ones i've used here has worked well enough.

# In[1051]:


# Here i set the best values for parameters i've already found.
best_n_estimators = 181
best_max_depth= 11

# Stores the best configuration in the loop
best_accuracy = 0
best_config = {}

#looping over different values for the next parameter to find, which here is min_samples split
parameters_to_try= [n for n in range(2,30,3)]

for parameter in parameters_to_try:
    #Storing accuracies for different random states
    accuracies = []

    # Iterate over a range of random states for train-test split
    for r in range(1, 100, 10):

        # Splitting data with varying random states, to avoid overfitting
        #also only using the selected features
        X_train, X_test, y_train, y_test = train_test_split(X[best_features], y, test_size=0.2, random_state=r)

        # Initialize and train the RandomForest model
        #Using the best paramters that i've found earlier with the same code
        #using different values for min_samples_split to find the new best one in this case
        model = RandomForestClassifier(random_state=42, criterion="gini", n_estimators=best_n_estimators, max_depth= best_max_depth, min_samples_split= parameter)
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    # Calculate the average accuracy across all splits
    average_accuracy = np.mean(accuracies)

    # Update the best configuration for new parameter if necessary
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_parameter= parameter
        best_config = {
            'n_estimators': best_n_estimators,
            'max_depth': best_max_depth,
            'random_state': 42,
            'criterion': "gini",
            'min_samples_split': best_parameter #store best new parameter
            #if im trying to find more paramters:
            #'min_samples_split': best_mms 
            #'new_parameter': best_parameter
        }

# Print the best configuration and its accuracy
print("Best Configuration:")
print(best_config)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")


# In[1053]:


# Just checking what parameters i ended up with
best_config


# In[1054]:


# Create a new RandomForestClassifier with the best configuration
final_model= RandomForestClassifier(**best_config)

#Training model with the whole dataset, but only with the selected features
final_model.fit(X[best_features], y)

#loading up the test-set
x_val= pd.read_csv("test.csv",index_col=0 )
#Selecting only the best feautres
x_val_selected = x_val[best_features]

#predicting
predictions= final_model.predict(x_val_selected).astype(int)


# In[1055]:


# Create a DataFrame with predictions
predictions_df = pd.DataFrame(predictions, columns=['Edible'])
predictions_df.index.name = 'index'

# Save to a CSV file
predictions_df.to_csv('predictions.csv')

