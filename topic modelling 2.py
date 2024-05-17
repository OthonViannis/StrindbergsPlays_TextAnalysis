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
directory = "C:/Users/viann/PycharmProjects/pythonProject1/venv/Translations"

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
##############################################################################
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

