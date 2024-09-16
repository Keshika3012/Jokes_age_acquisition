import pandas as pd
import json


#adultjokes
# Load the dataset
thejokecafe_data = pd.read_csv('/Users/keshikaa/Downloads/thejokecafe.csv')

# Remove duplicates
#thejokecafe_data = thejokecafe_data.drop_duplicates(subset='Joke')

# Remove irrelevant data (if any)
#thejokecafe_data = thejokecafe_data.dropna(subset=['Joke'])

# Ensure consistent format
#thejokecafe_data['Joke'] = thejokecafe_data['Joke'].str.lower()  # Convert to lowercase
#thejokecafe_data['Joke'] = thejokecafe_data['Joke'].str.replace('[^\w\s]', '')  # Remove punctuation

# Save the cleaned data (optional)
thejokecafe_data.to_csv('cleaned_thejokecafe.csv', index=False)

#teenjokes
# Load the dataset
reddit_cleanjokes_data = pd.read_csv('/Users/keshikaa/Downloads/reddit-cleanjokes.csv')

# Remove duplicates
#reddit_cleanjokes_data = reddit_cleanjokes_data.drop_duplicates(subset='Joke')

# Remove irrelevant data (if any)
#reddit_cleanjokes_data = reddit_cleanjokes_data.dropna(subset=['Joke'])

# Ensure consistent format
#reddit_cleanjokes_data['Joke'] = reddit_cleanjokes_data['Joke'].str.lower()  # Convert to lowercase
#reddit_cleanjokes_data['Joke'] = reddit_cleanjokes_data['Joke'].str.replace('[^\w\s]', '')  # Remove punctuation

# Save the cleaned data (optional)
reddit_cleanjokes_data.to_csv('cleaned_reddit_cleanjokes.csv', index=False)

#kidsjokes
# Load the dataset
kidsjokes_data = pd.read_csv('/Users/keshikaa/Downloads/kids_jokes-2.csv', sep=';', header=None)
#print(kidsjokes_data.head())
# Remove duplicates

#kidsjokes_data = kidsjokes_data.drop_duplicates(subset='Joke')

# Remove irrelevant data (if any)
#kidsjokes_data = kidsjokes_data.dropna(subset=['Joke'])

# Ensure consistent format
#kidsjokes_data['Joke'] = kidsjokes_data['Joke'].str.lower()  # Convert to lowercase
#kidsjokes_data['Joke'] = kidsjokes_data['Joke'].str.replace('[^\w\s]', '')  # Remove punctuation

#STEP 2
# Save the cleaned data (optional)
kidsjokes_data.to_csv('cleaned_kidsjokes.csv', index=False)


# Load the cleaned datasets
thejokecafe_cleaned = pd.read_csv('cleaned_thejokecafe.csv')
reddit_cleanjokes_cleaned = pd.read_csv('cleaned_reddit_cleanjokes.csv')
kidsjokes_data_cleaned = pd.read_csv('cleaned_kidsjokes.csv')

# Combine the datasets
combined_data = pd.concat([thejokecafe_cleaned, reddit_cleanjokes_cleaned,kidsjokes_data_cleaned], ignore_index=True)

# Remove any duplicates that may have been introduced during combination
combined_data = combined_data.drop_duplicates(subset='Joke')

# Save the combined dataset (optional)
combined_data.to_csv('combined_cleaned_jokes.csv', index=False)


import pandas as pd

# Load the datasets
df1 = pd.read_csv('cleaned_kidsjokes.csv') # Dataset 1
df2 = pd.read_csv('cleaned_reddit_cleanjokes.csv') # Dataset 2 
df3 = pd.read_csv('cleaned_thejokecafe.csv') # Dataset 3

# Add a label column to each dataset
df1['label'] = -1
df2['label'] = 0
df3['label'] = 1

# Concatenate the datasets
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the merged dataset to a new file
merged_df.to_csv('merged_dataset.csv', index=False)

print("Datasets merged successfully!")

#embedding

from gensim.models import Word2Vec
import pandas as pd

# Load your merged dataset
merged_data = pd.read_csv('merged_dataset.csv')

# Convert the 'Joke' column to strings
merged_data['Joke'] = merged_data['Joke'].astype(str)

# Tokenize your text
tokenized_text = [text.split() for text in merged_data['Joke']]

# Train the Word2Vec model
model = Word2Vec(tokenized_text, min_count=1)

# Get the embedding for a word
word_embedding = model.wv['word']

#print(tokenized_text)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume that X is a list of strings (documents) and y is a vector of labels
vectorizer = CountVectorizer()
X = merged_data['Joke'].tolist()
X = vectorizer.fit_transform(X)
y = merged_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
print(classification_report(y_test, y_pred))
# Assume that merged_data is your merged dataset and 'Joke' is the column that contains the text data


