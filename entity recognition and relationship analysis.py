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
expressionist_directory = "C:/Users/viann/PycharmProjects/pythonProject1/venv/Expressionism"

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
