import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#reading train-file
df= pd.read_csv("train.csv",index_col=0 )

#cleaning
df_cleaned= df.fillna(df.median())
df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['pH'] < 0].index)
df_cleaned= df_cleaned.drop(df_cleaned[df_cleaned["Acoustic Firmness Index"] > 50].index)
df_cleaned= df_cleaned.drop(df_cleaned[df_cleaned["Odor index (a.u.)"] > 85].index)

# Removing the target column from the DataFrame and uses the rest as features, Selcts Edible as target columns 
X = df_cleaned.drop('Edible', axis=1)  
y = df_cleaned['Edible']

# Creating RandomForestClassifier with the best configuration
final_model = RandomForestClassifier(random_state= 42, max_depth= 35, min_samples_split= 10, n_estimators=150)
# Training the classifier
final_model.fit(X, y)

#predicion
x_val= pd.read_csv("test.csv",index_col=0 )
predictions= final_model.predict(x_val).astype(int)

# Create a DataFrame with predictions
predictions_df = pd.DataFrame(predictions, columns=['Edible'])
predictions_df.index.name = 'index'
# Save to a CSV file
predictions_df.to_csv('predictions.csv')

#0.92592 is the best prediction from kaggle

"""
I'm not sure if i was supposed to explain more in this code, but more comments is in the notebook!
More explanation about my best classifier is under "Final Evaluation" in the notebook!" 
"""