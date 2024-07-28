# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:12:24 2023

@author: Kristian Røhne
"""

#%% Task 1

import pandas as pd

def generate_party_table(file_name, num_parties_to_include= None):
    # Read the file
    df= pd.read_csv(file_name, encoding='utf-8', header= 0, sep= ";")
     
    # Count the total number of votes for each party across all districts   
    party_votes = df.groupby("Partikode")["Antall stemmer totalt"].sum().reset_index()

    # Calculate the percentage of the votes of each party, with two decimals. 
    total_votes = party_votes['Antall stemmer totalt'].sum()
    party_votes['Stemmer i prosent%'] = (party_votes['Antall stemmer totalt'] / total_votes * 100).round(2)
    # Mark the parties that received at least 4% of the vote.
    party_votes['Over 4%?'] = party_votes['Stemmer i prosent%'].apply(lambda x: 'Yes' if x >= 4 else 'No')

    # Sort the results by the total number of votes in descending order
    party_votes_sorted = party_votes.sort_values(by='Antall stemmer totalt', ascending=False)
    
    # If num_parties_to_include is None or not provided, include all parties
    if num_parties_to_include is None: party_table = party_votes_sorted
    # Limit the table to the specified number of parties
    else: party_table = party_votes_sorted.head(num_parties_to_include)

    #Display the table with party codes, total votes, percentage of votes, and threshold
    print(party_table.to_string(index=False))
    print(" ")


#Trying the function with 3 parties
generate_party_table("2021-09-14_party distribution_1_st_2021.csv", 3)
#Trying the function with 7 parties
generate_party_table("2021-09-14_party distribution_1_st_2021.csv", 7)
#Trying the function with all parties 
generate_party_table("2021-09-14_party distribution_1_st_2021.csv")

#%% Task 2

import re


sentences = [
    "Alice and Bob are friends.",
    "John and Mary went to the park.",
    "David and Sarah enjoyed the movie.",
]

# Regular expression pattern to capture names
pattern = r'\b[A-Z][a-z]*\b'
text= f"{'Friendships':^20}"


def printing_namegroups(sentences):
    print(text)
    print("-"*len(text))
    for sentence in sentences:
        names= re.findall(pattern, sentence)
        print(f"{names[0]:>10} - {names[1]:<10}")
        
printing_namegroups(sentences)
