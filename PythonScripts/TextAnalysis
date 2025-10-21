import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

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

# Define the directory paths for the two subcorpora
first_directory = "C:/Users/viann/PycharmProjects/pythonProject1/venv/Translations"
second_directory = "C:/Users/viann/PycharmProjects/pythonProject1/venv/Expressionism"

# Initialize empty lists to store preprocessed documents for each subcorpus
first_preprocessed_documents = []
second_preprocessed_documents = []

# Function to preprocess documents in a directory
def preprocess_documents(directory):
    preprocessed_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                if text.strip():
                    preprocessed_text = preprocess_text(text)
                    preprocessed_docs.append(preprocessed_text)
    return preprocessed_docs

# Preprocess documents for the first subcorpus
first_preprocessed_documents = preprocess_documents(first_directory)
# Preprocess documents for the second subcorpus
second_preprocessed_documents = preprocess_documents(second_directory)

# Combine preprocessed documents for both subcorpora
all_preprocessed_documents = first_preprocessed_documents + second_preprocessed_documents

# Create a Dictionary object from the preprocessed documents
dictionary = Dictionary(all_preprocessed_documents)

# Filter out tokens that appear in less than `no_below` documents or more than `no_above` documents
# Keep only the first `keep_n` most frequent tokens
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

# Convert each document into a bag-of-words representation
corpus = [dictionary.doc2bow(doc) for doc in all_preprocessed_documents]

# Set the number of topics
num_topics = 5

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Get the topic distributions for each document
topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]

# Convert topic distributions to arrays for easier manipulation
topic_distributions_array = np.zeros((len(topic_distributions), num_topics))
for i, doc_topics in enumerate(topic_distributions):
    for topic_id, prob in doc_topics:
        topic_distributions_array[i][topic_id] = prob

# Use t-SNE to reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=10)  # Adjust perplexity as needed
topic_embeddings = tsne.fit_transform(topic_distributions_array)

# Separate the embeddings for each subcorpus
first_subcorpus_embeddings = topic_embeddings[:len(first_preprocessed_documents)]
second_subcorpus_embeddings = topic_embeddings[len(first_preprocessed_documents):]

# Plot scatter plots for each subcorpus
plt.figure(figsize=(10, 6))
plt.scatter(first_subcorpus_embeddings[:, 0], first_subcorpus_embeddings[:, 1], color='green', label='Translations')
plt.scatter(second_subcorpus_embeddings[:, 0], second_subcorpus_embeddings[:, 1], color='purple', label='Expressionism')
plt.title('Topic Distribution Comparison')
plt.xlabel('tt-SNE Similarity Dimension')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
# Print the t-SNE component values for each subcorpus
print("Translations t-SNE Component 1:")
print(first_subcorpus_embeddings[:, 0])
print("Translations t-SNE Component 2:")
print(first_subcorpus_embeddings[:, 1])
print("Expressionism t-SNE Component 1:")
print(second_subcorpus_embeddings[:, 0])
print("Expressionism t-SNE Component 2:")
print(second_subcorpus_embeddings[:, 1])
