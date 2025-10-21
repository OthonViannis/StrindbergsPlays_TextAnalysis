(*View the full project on my personal website: https://othonviannis.com/#strindbergpython)

**Literary Text Analysis: Mapping Strindberg - Data, Drama, and the Human Psyche**
This project explores the evolution of Swedish playwright August Strindberg's works from naturalism to expressionism, using digital humanities tools to uncover psychological 
and sociocultural patterns in 29 plays. By analyzing a corpus of naturalistic (realistic, social-focused) and expressionist (symbolic, emotional) plays, 
the work examines how Strindberg's personal crises influenced his themes, blending psychoanalytic (inner conflicts) and constructivist (societal norms) perspectives.

**Project Overview**
The analysis delves into Strindberg's dramatic oeuvre to reveal the latent semantic structures and emotional tones that define his naturalistic and expressionist phases. 
Through topic modeling, named entity recognition (NER), and sentiment analysis, the project illuminates the psychosocial landscapes in his plays, highlighting shifts from social realism 
to subconscious symbolism that mirror his life events, such as the Inferno crisis.

**Objectives**
The primary aims are to map major thematic flows in Strindberg's plays from 1872 to 1912, identify key entities and relationships that characterize each style, 
and gauge emotional tones to link them with biographical milestones. This demonstrates how data-driven methods can enrich literary studies, bridging humanities with computational analysis.

**Data & Methods**
The corpus consists of 29 plays, divided into naturalistic and expressionist sub-corpora. Topic modeling was performed using Python's Gensim library and LDA algorithm, 
identifying 5 topics per style. Naturalistic plays emphasize family dynamics and social issues (e.g., "marriage," "master"), reflecting 19th-century realism, while expressionist plays focus 
on existential motifs (e.g., "prince," "witch"), symbolizing inner turmoil.

Named Entity Recognition (NER) analyzed entities like locations, organizations, and persons with normalized frequencies. Expressionist plays show broader diversity (e.g., more GPE like "Jerusalem"), 
indicating symbolic exploration. A co-occurrence analysis constructed network graphs via matplotlib, revealing character connections and power dynamics.

Sentiment analysis applied VADER and TextBlob to gauge emotional tones. Naturalistic plays have a balanced sentiment (score 0.0730), while expressionist ones are more variable (0.0592). Timeline mapping to Strindberg's life events (e.g., Inferno crisis) links low scores to personal struggles, like in "The Stronger" (post-divorce).
Accompanying the analysis is a video timeline visualizing Strindberg’s life and creative journey. Using QGIS for mapping and DaVinci Resolve for editing, the video traces his residences, major publications, and key life events—blending spatial storytelling with literary data.
Key Visualizations

LDA Model Python visualization for topic distributions.
Bar Plots for Naturalistic and Expressionist top words.
Network Graphs for NER relationships in both styles.

**Insights**
BMI (r = 0.97) is the dominant predictor of diabetes risk, with poor physical and mental health adding modest effects. The project merges computational text analysis with visual storytelling, demonstrating how data-driven methods can illuminate the emotional and intellectual evolution of a writer. By combining literary study with digital mapping and video, it bridges humanities research with creative digital visualization.
Tools & Techniques

Python: Gensim (LDA topic modeling), pyLDAvis (topic visualization), matplotlib (bar plots and network graphs).
VADER and TextBlob for sentiment analysis.
QGIS for mapping, DaVinci Resolve for video timeline.
