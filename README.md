# DASHBOARD-DEVELOPMENT

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : PRATHAMESH MURKUTE

*INTERN ID* : CT06DF2317

*DOMAIN* : DATA ANALYTICS

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

## Sentiment Analysis Using NLP – Walmart Product Reviews Dataset

# Overview
This project is part of an internship task focused on applying Natural Language Processing (NLP) techniques to perform sentiment analysis on customer reviews. The dataset used is the Walmart Product Reviews Dataset, which was downloaded from Kaggle and processed in Google Colab using Python. The goal was to classify sentiments (positive or negative) from text reviews using machine learning models.

# Objectives
To preprocess raw textual data using NLP techniques.

To transform the text into a machine-understandable format using feature extraction methods.

To build a sentiment classification model using supervised learning algorithms.

To visualize data insights and evaluate model performance.

# Tools & Technologies Used
Platform: Google Colab

Language: Python

# Libraries:

pandas, numpy: For data manipulation

matplotlib, seaborn: For visualization

nltk, re: For NLP preprocessing

scikit-learn: For model building and evaluation

# Dataset
The dataset was downloaded from Kaggle and contains Walmart product reviews, including:

Review Text

Rating

Title

Review Summary

For the sentiment analysis task, reviews were labeled based on ratings:

Ratings ≥ 4 were labeled positive

Ratings ≤ 2 were labeled negative

Rating 3 (neutral) was either excluded or handled based on design choice

# Data Preprocessing
To clean and prepare the textual data:

Removed special characters, punctuations, and numbers using regex (re)

Converted all text to lowercase for normalization

Tokenized the sentences

Removed stopwords using nltk

Applied stemming or lemmatization

After preprocessing, the data was vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) or CountVectorizer to convert textual data into numerical format for model training.

Model Implementation
Several classification models were evaluated:

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

The dataset was split into training and testing sets (typically 80-20 ratio). Models were trained on the training data and predictions were made on the test set.

Evaluation Metrics
Model performance was evaluated using the following metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Visualization tools such as seaborn’s heatmap were used to display confusion matrices, and bar plots were generated to compare model performance.

# Results
The Logistic Regression and Naive Bayes models performed well, with accuracy scores over 85% in most trials. Text preprocessing and proper vectorization played a key role in achieving good results.

# Conclusion
The project successfully demonstrated the application of NLP techniques and machine learning models to analyze customer sentiments. By cleaning, transforming, and classifying textual data, we were able to derive meaningful insights and evaluate product perception. This type of analysis can help Walmart and similar retailers in improving product quality, customer satisfaction, and service delivery.

# Future Work
Use of deep learning models like LSTM or BERT for more advanced performance.

Incorporating more granular sentiment classes (e.g., very positive, neutral, very negative).

Creating a web dashboard to monitor sentiment trends in real-time.

# Technologies Used
•	Python 3.10+
•	Google Colab (for development and execution)
•	Libraries:
o	pandas
o	numpy
o	matplotlib (if visualization is included)

# How to Run
1.	Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git
Open the .py script in Google Colab or Jupyter Notebook.
Upload the TripAdvisor dataset (tripadvisor_hotel_reviews.csv) if not already available.
Run all cells to view the analysis and results.

# Author
Name: Prathamesh Murkute
Task: PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING - Task 3
Platform: Google Colab

# License
This project is licensed under the MIT License.
