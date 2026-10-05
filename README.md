**British Airways Customer Reviews: Sentiment Analysis & Booking Prediction**

*Two-part project analysing British Airways customer feedback and predicting booking completion from flight-search behaviour. Built as part of my MSc Data Science coursework at Bournemouth University.*


**Overview**


Airlines generate huge volumes of unstructured customer feedback and structured booking data, but struggle to turn either into actionable signal. This project tackles both sides: understanding what customers are saying (Part 1) and predicting who will complete a booking (Part 2).


**Part 1**: Web Scraping & Sentiment Analysis

**Goal**: Collect and analyse customer reviews to understand sentiment drivers.


**Method**:

•	Scraped ~1,000 customer reviews from Airline Quality using BeautifulSoup and requests

•	Cleaned text: removed verification tags, expanded contractions, stripped URLs and special characters

•	Tokenised and lemmatised text with NLTK, with custom stopword tuning (kept "not" to preserve negative sentiment signal)

•	Generated a word cloud to visualise frequent terms

•	Scored sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner), classifying each review as Positive, Negative, or Neutral based on compound score thresholds


**Output**: Sentiment distribution across the review set, visualised as a pie chart.



**Part 2**: Predicting Booking Completion

**Goal**: Predict whether a flight search converts into a completed booking, using customer and flight attributes.


**Method**:

•	Exploratory analysis of booking completion against trip type and sales channel

•	Feature selection via Mutual Information to rank which variables (route, booking origin, flight duration, extras requested, lead time, etc.) carry the most predictive signal

•	Preprocessing: label encoding for categorical features, min-max scaling, 70/30 train-test split

•	Trained a Random Forest Classifier (150 estimators) to predict booking_complete

•	Evaluated using precision, recall, and F-score


**Tech Stack**

Python · pandas · NumPy · BeautifulSoup · NLTK (VADER, WordNet lemmatizer) · scikit-learn · seaborn · matplotlib · WordCloud


**Files**

•	01_scraping_sentiment_analysis.ipynb — Part 1: scraping, cleaning, sentiment scoring

•	02_booking_prediction.ipynb — Part 2: feature selection, modelling, evaluation


**Notes**

This was completed as a structured two-task coursework exercise, so Part 2 uses a provided customer_booking.csv dataset rather than the scraped Part 1 data. Both parts share a common theme: extracting predictive/explanatory signal from airline customer data.

