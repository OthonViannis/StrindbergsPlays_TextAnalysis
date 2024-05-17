import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Directory containing your preprocessed text files for Expressionist plays
expressionism_directory = 'C:/Users/viann/PycharmProjects/pythonProject1/venv/Expressionism'

# Directory containing your preprocessed text files for Naturalistic plays
naturalism_directory = 'C:/Users/viann/PycharmProjects/pythonProject1/venv/Translations'

# Initialize an empty list to store the contents of all cleaned text files for Expressionist plays
expressionism_texts = []

# Loop through each file in the Expressionist plays directory
for filename in os.listdir(expressionism_directory):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Read the contents of the file
        with open(os.path.join(expressionism_directory, filename), 'r', encoding='utf-8') as file:
            # Read the contents and append them to the list
            contents = file.read()
            expressionism_texts.append(contents)

# Concatenate the contents into a single string for Expressionist plays
all_expressionism_texts = ' '.join(expressionism_texts)

# Initialize an empty list to store the contents of all cleaned text files for Naturalistic plays
naturalism_texts = []

# Loop through each file in the Naturalistic plays directory
for filename in os.listdir(naturalism_directory):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Read the contents of the file
        with open(os.path.join(naturalism_directory, filename), 'r', encoding='utf-8') as file:
            # Read the contents and append them to the list
            contents = file.read()
            naturalism_texts.append(contents)

# Concatenate the contents into a single string for Naturalistic plays
all_naturalism_texts = ' '.join(naturalism_texts)

# Tokenize the text and count word frequencies for Expressionist plays
expressionism_word_freq = Counter(all_expressionism_texts.split())
#
# # Tokenize the text and count word frequencies for Naturalistic plays
naturalism_word_freq = Counter(all_naturalism_texts.split())
#
# # Define a list of tokens to be excluded from the analysis
exclude_tokens = ["n't", "--", "'s", "—", "’", "yes", "...."]
#
# # Filter out the excluded tokens from the list of most common words
expressionism_word_freq_filtered = {word: freq for word, freq in expressionism_word_freq.items() if word not in exclude_tokens}
naturalism_word_freq_filtered = {word: freq for word, freq in naturalism_word_freq.items() if word not in exclude_tokens}
#
# # Convert the filtered dictionaries back to Counter objects
expressionism_word_freq_filtered = Counter(expressionism_word_freq_filtered)
naturalism_word_freq_filtered = Counter(naturalism_word_freq_filtered)
#
# # Display the filtered list of most common words
print("Filtered Most Common Words in Expressionist plays:")
print(expressionism_word_freq_filtered.most_common(10))
print("\nFiltered Most Common Words in Naturalistic plays:")
print(naturalism_word_freq_filtered.most_common(10))
#
# # Calculate sentiment scores for the filtered text data
expressionist_sentiment_score_filtered = TextBlob(all_expressionism_texts).sentiment.polarity
naturalistic_sentiment_score_filtered = TextBlob(all_naturalism_texts).sentiment.polarity
#
# # Display the sentiment scores for the filtered text data
print("\nSentiment Score for Expressionist plays (Filtered):", expressionist_sentiment_score_filtered)
print("Sentiment Score for Naturalistic plays (Filtered):", naturalistic_sentiment_score_filtered)
###############################################################################
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import re

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Directory containing your preprocessed text files for Expressionist plays
expressionism_directory = 'C:/Users/viann/PycharmProjects/pythonProject1/venv/Expressionism'

# Directory containing your preprocessed text files for Naturalistic plays
naturalism_directory = 'C:/Users/viann/PycharmProjects/pythonProject1/venv/Translations'

# Function to read text files and perform sentiment analysis
def analyze_sentiment(directory):
    sentiment_scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Additional preprocessing steps
                text = text.lower()  # Convert text to lowercase
                text = re.sub(r'\bnot\b', 'not_', text)  # Add underscore to handle negations
                text = re.sub(r'(:\)|:\(|:-\)|:-\()', '', text)  # Remove emoticons
                text = re.sub(r'\b(very|extremely|highly)\b', '', text)  # Remove intensifiers
                # Analyze sentiment
                scores = analyzer.polarity_scores(text)
                sentiment_scores.append(scores)
    return sentiment_scores

# Perform sentiment analysis for Expressionist plays
expressionism_sentiment_scores = analyze_sentiment(expressionism_directory)

# Perform sentiment analysis for Naturalistic plays
naturalism_sentiment_scores = analyze_sentiment(naturalism_directory)

# Print sentiment scores for Expressionist plays
print("Sentiment Scores for Expressionist plays:")
for i, scores in enumerate(expressionism_sentiment_scores, start=1):
    print(f"Play {i}: {scores}")

# Print sentiment scores for Naturalistic plays
print("\nSentiment Scores for Naturalistic plays:")
for i, scores in enumerate(naturalism_sentiment_scores, start=1):
    print(f"Play {i}: {scores}")

######################################
import pandas as pd
import matplotlib.pyplot as plt
#
# # Load your dataset
data = pd.read_csv('C:/Users/viann/PycharmProjects/pythonProject1/venv/StrindbergsPlayDataset.csv')
#
#
# # Ensure the date column is in a proper datetime format (if it's just a year, adjust the format accordingly)
data['date'] = pd.to_datetime(data['date'], format='%Y')  # Adjust the format if your date data is different

# # Sort data by date for chronological plotting
data = data.sort_values(by='date')

# Plotting the sentiment scores over time
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['compound'], marker='o', linestyle='-', color='blue')  # Ensure 'compound' matches your sentiment score column name
plt.title('Sentiment Analysis Over Time')
plt.xlabel('Year')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()
