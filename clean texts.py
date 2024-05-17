import os
import spacy
from collections import Counter

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_sm")

# Directory containing your cleaned text files
directory = 'C:/Users/viann/PycharmProjects/pythonProject1/venv/Expressionism'

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

###################################################
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