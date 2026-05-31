# Fake News Detection Project



The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not. The distribution of fake news can potentially have highly adverse effects on people and culture. This project involves building and training a model to classify news as fake news or not using a diverse dataset of news articles. We have used four techniques to determine the results of the model.

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boost Classifier**
4. **Random Forest Classifier**

## Project Overview

Fake news has become a significant issue in today's digital age, where information spreads rapidly through various online platforms. This project leverages machine learning algorithms to automatically determine the authenticity of news articles, providing a valuable tool to combat misinformation.

## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

## System Requirements 

Hardware :
1. 4GB RAM
2. i3 Processor
3. 500MB free space

Software :
1. Anaconda
2. Python

## Dependencies

Before running the code, make sure you have the following libraries and packages installed:

- Python 3
- Scikit-learn
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Regular Expression

You can install these dependencies using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn 
pip install re 
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/AryanKushwaha01/Fake-News-Detection.git
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Execute the Jupyter Notebook or use the newly added modular engine (`fake_news_detector.py`) to train and test models:
   ```bash
   python fake_news_detector.py
   ```

4. Alternatively, launch the **Veritas AI Premium Web UI Suite** for an interactive, glassmorphic dashboard:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000` to start analyzing claims, contrasting ensemble outputs, and evaluating TF-IDF feature weights in real time.

## Veritas AI Premium Web UI Features

- **Consensus Rating**: Real-time gauge displaying aggregated fake scores computed from all four active classifiers.
- **Ensemble Contrast Panels**: Side-by-side indicators showing prediction confidence bars for Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest.
- **TF-IDF Keyword Highlighter**: Extract and display unique mathematical features/vocabulary from entered news texts in interactive neon tags.
- **Prediction Session Ledger**: Built-in interactive historical prediction table powered by LocalStorage for logging past results.
- **Instant Cycles & Clears**: Cycle-paste true and fabricated sample news articles directly from the dashboard to run instant, hands-on sanity checks.

## Results

We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files and can be verified via the interactive terminal engine or the Web UI.

## Model Deployment

Once you are satisfied with the performance of a particular classifier, you can deploy it in a real-world application or run the integrated Flask web server (`app.py`) directly as an interface for news forensics.
---

