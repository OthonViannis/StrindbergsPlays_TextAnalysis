#########################################ThematicAnalysis################################################
import os
import spacy
from collections import Counter

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_sm")

# Directory containing your cleaned text files
directory = 'C:/path/to/your/directory'

# Initialize an empty list to store the contents of all cleaned text files
texts = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Read the contents of the file
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            # Read the contents and append them to the list
            contents = file.read()
            texts.append(contents)

# Concatenate the contents into a single string
all_texts = ' '.join(texts)

# Process the text in smaller batches
batch_size = 1000000  # Set the batch size based on your system's memory constraints
batches = [all_texts[i:i+batch_size] for i in range(0, len(all_texts), batch_size)]

# Tokenize the text in each batch and count word frequencies
word_freq = Counter()
for batch in batches:
    doc = nlp(batch)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text != "`"]
    word_freq.update(tokens)

#Print the most common words
print("Most common words:")
for word, freq in word_freq.most_common():
    print(word, freq)

#Perform further analysis and visualization as needed
####################################

#Function to process text in batches
def process_text(text):
    doc = nlp(text)
    # Filter out stop words and punctuation
    filtered_words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    # Count word frequencies
    word_freq = Counter(filtered_words)
    return word_freq

# Concatenate the contents into a single string
all_texts = ' '.join(texts)

# Split the text into smaller segments
chunk_size = 1000000  # Adjust the chunk size as needed
chunks = [all_texts[i:i+chunk_size] for i in range(0, len(all_texts), chunk_size)]

# Process each chunk and combine the results
word_freq_combined = Counter()
for chunk in chunks:
    word_freq_chunk = process_text(chunk)
    word_freq_combined += word_freq_chunk

specific_words = ['stranger', 'king', 'lady', 'mother', 'daughter', 'child', 'good',
             'man', 'time', 'life', 'prince', 'master', 'woman', 'people',
             'father', 'judge', 'god', 'wife', 'lawyer', 'believe',
             'friend', 'lord', 'major', 'sergeant',
             'home', 'light', 'poet', 'moment', 'feel',
            'heart', 'queen', 'fire', 'confessor', 'voice',
             'law', 'dead', 'church', 'end', 'peace', 'evil', 'beggar',
             'girl', 'sun', 'fall',
             'death', 'lost', 'human', 'tempter', 'die', 'brother',
             'heaven', 'strange', 'silence', 'mind', 'suffering']

# Define your themes and associated words
theme_associated_words = {
    'love': {'NOUN': ['love', 'romance', 'affection'], 'VERB': ['love', 'adore', 'cherish']},
    'conflict': {'NOUN': ['war', 'battle', 'struggle'], 'VERB': ['fight', 'struggle', 'clash']},
    'identity': {'NOUN': ['self', 'identity', 'individuality'], 'ADJ': ['personal', 'unique', 'distinct']},
    'psychology': {'NOUN': ['mind', 'psychology', 'consciousness'], 'ADJ': ['psychological', 'mental', 'emotional']},
    'nature': {'NOUN': ['nature', 'earth', 'environment'], 'ADJ': ['natural', 'environmental', 'organic']},
    'fate': {'NOUN': ['fate', 'destiny', 'fortune'], 'VERB': ['decide', 'determine', 'destine']},
    'gender': {'NOUN': ['gender', 'sexuality', 'femininity'], 'ADJ': ['gendered', 'sexual', 'feminine']},
    'spirituality': {'NOUN': ['spirituality', 'soul', 'faith'], 'ADJ': ['spiritual', 'divine', 'sacred']},
    'isolation': {'NOUN': ['isolation', 'solitude', 'seclusion'], 'ADJ': ['isolated', 'solitary', 'secluded']},
    'society': {'NOUN': ['society', 'community', 'culture'], 'ADJ': ['social', 'cultural', 'communal']},
    'madness': {'NOUN': ['madness', 'insanity', 'lunacy'], 'ADJ': ['mad', 'insane', 'lunatic']},
    'class': {'NOUN': ['class', 'social class', 'caste'], 'ADJ': ['class-based', 'social', 'economic']},
    'symbolism': {'NOUN': ['symbolism', 'symbol', 'metaphor'], 'ADJ': ['symbolic', 'figurative', 'representative']},
    'death': {'NOUN': ['death', 'mortality', 'demise'], 'ADJ': ['deadly', 'mortal', 'fatal']},
    'family': {'NOUN': ['family', 'relatives', 'kin'], 'ADJ': ['family', 'kin', 'related']},
    'time': {'NOUN': ['time', 'moment', 'era'], 'ADJ': ['temporary', 'timely', 'chronological']},
    'dreams': {'NOUN': ['dream', 'fantasy', 'vision'], 'VERB': ['dream', 'imagine', 'fantasize']},
}


    # Add more themes and associated words for different parts of speech as needed

# Categorize words into themes based on their relevance
theme_freq = {theme: 0 for theme in theme_associated_words}
for word, freq in word_freq_combined.items():
    for theme, words in theme_associated_words.items():
        for pos, word_list in words.items():
            if word in word_list:
                theme_freq[theme] += freq

# Print the frequency of each theme
for theme, freq in theme_freq.items():
    print(f"Theme: {theme}, Frequency: {freq}")

#################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Common words with frequencies
common_words_data = {
    'word': ['stranger', 'king', 'lady', 'mother', 'daughter', 'child', 'good',
             'man', 'time', 'life', 'prince', 'master', 'woman', 'people',
             'father', 'judge', 'god', 'wife', 'lawyer', 'believe',
             'friend', 'lord', 'major', 'sergeant',
             'home', 'light', 'poet', 'moment', 'feel',
            'heart', 'queen', 'fire', 'confessor', 'voice',
             'law', 'dead', 'church', 'end', 'peace', 'evil', 'beggar',
             'girl', 'sun', 'fall',
             'death', 'lost', 'human', 'tempter', 'die', 'brother',
             'heaven', 'strange', 'silence', 'mind', 'suffering'],  # Added a placeholder word
    'frequency': [1373, 1079, 972, 742, 733, 644, 593, 571, 553,
                  544, 475, 441, 410, 393, 382, 360, 346, 303,
                  296, 291, 286, 239, 220,
                  219, 216, 214, 213, 212, 205, 200,
                  197, 193, 172, 171, 170, 170,
                  166, 161, 160, 158, 155, 155, 155, 152, 148,
                  147, 145, 134, 129, 127,
                  125, 125, 125, 116, 113]
}

print(len(common_words_data['word']))
print(len(common_words_data['frequency']))

# Create DataFrame
common_words_df = pd.DataFrame(common_words_data)

# Thematic Coding: Create a DataFrame with thematic labels for common words
thematic_coding = pd.DataFrame(columns=['word', 'theme'])

# Frequency Analysis: Plot the frequency of common words
plt.figure(figsize=(10, 6))
sns.barplot(data=common_words_df, x='word', y='frequency', color='purple')  # Set color to purple
plt.title('Frequency of Common Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Theme data for expressionistic plays
expressionistic_themes_data = {
    'Theme': ['love', 'conflict', 'identity', 'psychology', 'nature', 'fate', 'gender', 'spirituality',
              'isolation', 'society', 'madness', 'class', 'symbolism', 'death', 'family', 'time', 'dreams'],
    'Frequency': [575, 178, 56, 114, 174, 119, 2, 217, 18, 22, 41, 8, 14, 158, 181, 677, 217]
}

# Create DataFrame for expressionistic themes
expressionistic_themes_df = pd.DataFrame(expressionistic_themes_data)

# Sort expressionistic themes by frequency in descending order
expressionistic_themes_df_sorted = expressionistic_themes_df.sort_values(by='Frequency', ascending=False)

# Plotting the bar graph with reversed color palette
plt.figure(figsize=(10, 6))
reversed_purple_palette = sns.color_palette("Purples_r", n_colors=len(expressionistic_themes_df_sorted))
sns.barplot(data=expressionistic_themes_df_sorted, x='Theme', y='Frequency', palette=reversed_purple_palette)
plt.title('Frequency of Themes in Expressionist Plays')
plt.xlabel('Theme')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

#############################################TopicModelling#####################################################
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Define your preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    return filtered_text


# Define the directory containing the text files
directory = "C:/path/to/your/directory"

# Initialize an empty list to store preprocessed documents
preprocessed_documents = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)

        # Read the contents of the file
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

            # Ensure text is not empty
            if text.strip():
                # Preprocess the text
                preprocessed_text = preprocess_text(text)

                # Add the preprocessed text to the list of preprocessed documents
                preprocessed_documents.append(preprocessed_text)

# Now preprocessed_documents contains the preprocessed text of each document in your directory
# You can proceed to create the corpus variable using this preprocessed data

from gensim.corpora import Dictionary

# Create a Dictionary object from the preprocessed documents
dictionary = Dictionary(preprocessed_documents)

# Filter out tokens that appear in less than `no_below` documents or more than `no_above` documents
# Keep only the first `keep_n` most frequent tokens
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

# Convert each document into a bag-of-words representation
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

from gensim.models import LdaModel

# Set the number of topics
num_topics = 5

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print("Topic {}: {}".format(idx, topic))


import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Convert the LDA model into the pyLDAvis format
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(vis_data)


import matplotlib.pyplot as plt


# Function to visualize topics
def visualize_topics(lda_model, num_words=10):
    # Get the top words for each topic
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    # Create a bar plot for each topic
    for topic_id, words in topics:
        words = dict(words)
        plt.figure(figsize=(8, 6))
        plt.barh(list(words.keys()), list(words.values()), color='green')
        plt.gca().invert_yaxis()
        plt.xlabel('Word Probability')
        plt.title(f'Topic {topic_id}')
        plt.show()


# Visualize topics
visualize_topics(lda_model)

######################################EntityRecognition&RelationshipAnalysis###########################################
import os
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to read texts from a directory
def read_texts_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

# Function to analyze co-occurrences of entities
def co_occurrence_analysis(texts):
    entity_co_occurrences = defaultdict(lambda: defaultdict(int))
    for text in texts:
        doc = nlp(text)
        seen_entities = set()
        for token in doc:
            if token.pos_ in ["PROPN", "NOUN"] and token.text.isalpha():
                for other_token in seen_entities:
                    entity_co_occurrences[token.text][other_token] += 1
                    entity_co_occurrences[other_token][token.text] += 1
                seen_entities.add(token.text)
    return entity_co_occurrences

# Function to create a network graph from co-occurrences with a minimum threshold
def create_filtered_network_graph(co_occurrences, min_degree=5, min_weight=15, max_entities=40):
    G = nx.Graph()
    for entity, connections in co_occurrences.items():
        for connected_entity, weight in connections.items():
            if weight >= min_weight and len(G) < max_entities:  # Limit the number of entities
                G.add_edge(entity, connected_entity, weight=weight)
    return G

# Directory containing the text files for Expressionist plays
expressionist_directory = "C:/path/to/your/directory"

# Read texts from the directory
expressionism_texts = read_texts_from_directory(expressionist_directory)

# Analyze co-occurrences of entities in expressionist plays
Expressionist_co_occurrences = co_occurrence_analysis(expressionism_texts)

# Use updated functions for visualization
G_expressionist = create_filtered_network_graph(Expressionist_co_occurrences, max_entities=40)

# Define the specific words you want to highlight
highlighted_words = ["kindness", "church", "year", "time", "happiness"]

# Draw the network graph for Expressionist plays
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_expressionist)

# Draw nodes with purple color
nx.draw_networkx_nodes(G_expressionist, pos, node_size=3000, node_color="purple")

# Draw edges
nx.draw_networkx_edges(G_expressionist, pos)

# Draw labels with specific color for highlighted words
node_labels = {node: node if node in highlighted_words else "" for node in G_expressionist.nodes}
nx.draw_networkx_labels(G_expressionist, pos, labels=node_labels, font_size=10, font_weight="bold")

# Set font color for each label individually
for node, (x, y) in pos.items():
    if node in highlighted_words:
        plt.text(x, y, node, fontsize=10, color='red', fontweight='bold')
    else:
        plt.text(x, y, node, fontsize=10, color='black', fontweight='bold')

plt.title("Network: Expressionist Plays")
plt.show()

##############################################SentimentAnalysis#####################################################

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
#############################

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import re

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Directory containing your preprocessed text files for Expressionist plays
expressionism_directory = 'C:/path/to/your/directory'

# Directory containing your preprocessed text files for Naturalistic plays
naturalism_directory = 'C:/path/to/your/directory'

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
data = pd.read_csv('C:/path/to/your/directory')
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

##############################################ComparativeAnalysis#####################################################

import os
from collections import Counter


# Function to read texts from a directory
def read_texts_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts


# Function for text comparison analysis
def compare_texts(naturalistic_texts, expressionist_texts):
    # Tokenize and preprocess the text data (add your preprocessing steps here)

    # Calculate word frequency for Naturalistic plays
    naturalistic_word_freq = Counter()
    for text in naturalistic_texts:
        # Tokenize text and update word frequency
        words = text.split()  # Example tokenization, modify as needed
        naturalistic_word_freq.update(words)

    # Calculate word frequency for Expressionist plays
    expressionist_word_freq = Counter()
    for text in expressionist_texts:
        # Tokenize text and update word frequency
        words = text.split()  # Example tokenization, modify as needed
        expressionist_word_freq.update(words)

    # Highlighting: Identify significant differences in word frequency
    significant_differences = {word: expressionist_word_freq[word] - naturalistic_word_freq[word]
                               for word in expressionist_word_freq if
                               expressionist_word_freq[word] > 2 * naturalistic_word_freq[word]}

    # Summarization: Provide summary statistics
    total_naturalistic_words = sum(naturalistic_word_freq.values())
    total_expressionist_words = sum(expressionist_word_freq.values())
    common_words = set(naturalistic_word_freq.keys()) & set(expressionist_word_freq.keys())
    total_common_words = len(common_words)
    unique_naturalistic_words = len(naturalistic_word_freq) - total_common_words
    unique_expressionist_words = len(expressionist_word_freq) - total_common_words
    summary = {
        "Total Naturalistic Words": total_naturalistic_words,
        "Total Expressionist Words": total_expressionist_words,
        "Total Common Words": total_common_words,
        "Unique Naturalistic Words": unique_naturalistic_words,
        "Unique Expressionist Words": unique_expressionist_words
    }

    # Interactive Tools: Create interactive tools (not implemented in this example)

    # Narrative Explanation: Provide a narrative explanation
    explanation = """
    The analysis compared word frequencies between Naturalistic and Expressionist plays.
    Overall, Expressionist plays tend to have a higher frequency of emotionally charged words and symbolic language,
    reflecting the genre's focus on internal psychological states and abstract themes. Naturalistic plays, on the other hand,
    often emphasize realism and everyday language, resulting in a more balanced distribution of word frequencies.
    The highlighted words represent significant differences between the two genres, indicating potential areas of thematic divergence.
    """

    return significant_differences, summary, explanation


# Directory containing the text files for Naturalistic plays
naturalistic_directory = "C:/path/to/your/directory"

# Directory containing the text files for Expressionist plays
expressionist_directory = "C:/path/to/your/directory"

# Read texts from the directories
naturalistic_texts = read_texts_from_directory(naturalistic_directory)
expressionist_texts = read_texts_from_directory(expressionist_directory)

# Perform text comparison analysis
significant_differences, summary, explanation = compare_texts(naturalistic_texts, expressionist_texts)

# Print or visualize the results
print("Significant Differences:", significant_differences)
print("Summary:", summary)
print("Explanation:", explanation)
#################
import matplotlib.pyplot as plt

# Sentiment scores for Naturalistic and Expressionist plays
naturalistic_sentiment = [0.073, 0.085, 0.062, 0.078, 0.056, 0.068, 0.072, 0.067, 0.079, 0.061]  # Example sentiment scores for Naturalistic plays
expressionist_sentiment = [0.059, 0.071, 0.053, 0.064, 0.049, 0.055, 0.067, 0.058, 0.063, 0.052]  # Example sentiment scores for Expressionist plays

# Play numbers for Naturalistic and Expressionist plays
naturalistic_plays = list(range(1, len(naturalistic_sentiment) + 1))
expressionist_plays = list(range(1, len(expressionist_sentiment) + 1))

# Create a scatter plot
plt.scatter(naturalistic_plays, naturalistic_sentiment, label='Naturalistic', color='blue')
plt.scatter(expressionist_plays, expressionist_sentiment, label='Expressionist', color='red')

# Add labels and title
plt.xlabel('Play Number')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis by Play')
plt.legend()
plt.grid(True)

# Show plot
plt.show()


