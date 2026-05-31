import os
import re
import string
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# File Paths
FAKE_CSV = 'Fake.csv'
TRUE_CSV = 'True.csv'
MODEL_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'lr_model.pkl')
DT_MODEL_PATH = os.path.join(MODEL_DIR, 'dt_model.pkl')
GB_MODEL_PATH = os.path.join(MODEL_DIR, 'gb_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

def wordopt(text):
    """
    Cleans and preprocesses the input text.
    Removes HTML tags, URLs, punctuation, extra spaces, and digits.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)  # FIXED: Removed bytes literal b'' to prevent TypeError in Python 3
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def ensure_datasets_exist():
    """
    Checks if Fake.csv and True.csv exist. If not, generates a synthetic dataset
    so the pipeline remains executable and testable without raising FileNotFoundError.
    """
    if os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
        print("[SUCCESS] Found existing datasets 'Fake.csv' and 'True.csv'.")
        return False

    print("[WARNING] Datasets 'Fake.csv' or 'True.csv' not found.")
    print("[INFO] Generating synthetic placeholder datasets for demonstration...")
    
    # Synthetic True News
    true_data = {
        'title': [
            'President announces new economic policy',
            'Scientists discover new water source on Mars',
            'Global summit addresses climate change challenges',
            'Local library hosts annual reading festival',
            'Tech giant launches open source AI model',
            'Major medical breakthrough in cancer research announced',
            'Spacecraft successfully lands on asteroid',
            'Prime Minister signs historic trade agreement',
            'National park expands protected wildlife zone',
            'New education bill passed by congress with bipartisan support'
        ] * 50,
        'text': [
            'WASHINGTON (Reuters) - The federal government announced a series of sweeping economic reforms today aimed at curbing inflation and boosting job growth across tech and manufacturing sectors.',
            'PASADENA (Reuters) - NASA researchers have confirmed the discovery of liquid water reserves deep beneath the surface of Mars, using high-resolution radar data from orbiting spacecraft.',
            'GENEVA (Reuters) - World leaders gathered in Switzerland to sign a landmark treaty committing nations to faster carbon emission reductions over the next decade.',
            'BOSTON (Reuters) - The Boston Public Library welcomed over ten thousand visitors to its annual weekend reading festival, promoting literacy and supporting local independent authors.',
            'SAN FRANCISCO (Reuters) - Leading artificial intelligence company announced the release of its advanced neural network codebase under a fully permissive open-source license.',
            'LONDON (Reuters) - Clinical trials at Oxford have shown highly promising results for a new targeted immunotherapy, showing high efficacy rates in early-stage trials.',
            'TOKYO (Reuters) - The national space agency confirmed that its deep-space probe has successfully touched down on asteroid Ryugu and collected surface samples.',
            'BRUSSELS (Reuters) - Officials from the European Union signed a comprehensive free trade deal, lowering tariffs and standardizing regulatory frameworks.',
            'DENVER (Reuters) - The Department of the Interior has added over fifty thousand acres to the Rocky Mountain National Park to protect migratory routes of elk and sheep.',
            'WASHINGTON (Reuters) - Both parties came together to pass the educational funding bill, providing historic resources to public schools and state universities.'
        ] * 50,
        'subject': ['politics', 'science', 'worldnews', 'local', 'tech', 'science', 'science', 'worldnews', 'local', 'politics'] * 50,
        'date': ['May 20, 2026'] * 500
    }
    
    # Synthetic Fake News
    fake_data = {
        'title': [
            'Shocking! Secret alien base discovered in Antarctica',
            'Elixir of youth found hidden inside ancient pyramid',
            'Government planning to ban all physical currency next week',
            'Hollywood star reveals secret society controlling the weather',
            'Water-powered engine invented but suppressed by oil companies',
            'Unbelievable: Local politician caught speaking to time travelers',
            'Superfood cures all diseases instantly, health officials shocked',
            'Scientists prove the moon is actually hollow and made of metal',
            'Lost city of Atlantis found off the coast of Florida',
            'New law forces citizens to adopt robotic pets by end of year'
        ] * 50,
        'text': [
            'SHOCKING NEWS! Whistleblowers have leaked top secret military documents proving that a massive underground alien base is operating in Antarctica, completely hidden from the public eye.',
            'MUST READ: Explorers in Egypt have discovered an ancient chamber containing a glowing potion that reverses aging. Big pharma is trying to hide this discovery from you!',
            'The government is secretly preparing a complete shutdown of the banking system this coming Sunday. All physical paper cash will be declared illegal and confiscated.',
            'In a viral live-stream, a popular Hollywood actor exposed the secret climate control lasers operated by a global cabal that are causing summer snowstorms.',
            'An independent inventor has successfully run his car on standard tap water for six months. However, secret agents raided his lab and destroyed the blueprints.',
            'Exclusive footage shows a local senator entering a suspicious blue phone booth and disappearing, with sources claiming he is collaborating with future rulers.',
            'This simple kitchen ingredient will cure diabetes, cancer, and heart disease overnight. Doctors are furious that this cheap secret has been leaked online.',
            'Astronomers have leaked audio recordings from the Apollo mission showing hollow echoing sounds when landing, proving the moon is a giant metal hollow sphere.',
            'SATELLITE IMAGES reveal massive marble structures and pyramids under the ocean near Miami, confirming that the mythical lost continent of Atlantis has been found.',
            'An official decree will require every household to replace their domestic dogs and cats with automated robotic AI companions to conserve ecological resources.'
        ] * 50,
        'subject': ['conspiracy', 'mysteries', 'fakePolitics', 'entertainment', 'secrets', 'conspiracy', 'health', 'secrets', 'mysteries', 'fakePolitics'] * 50,
        'date': ['May 20, 2026'] * 500
    }

    pd.DataFrame(true_data).to_csv(TRUE_CSV, index=False)
    pd.DataFrame(fake_data).to_csv(FAKE_CSV, index=False)
    print("[SUCCESS] Synthetic 'Fake.csv' and 'True.csv' files successfully created!")
    return True

def train_and_save_models():
    """
    Runs the entire data loading, cleaning, training, and saving pipeline.
    """
    is_synthetic = ensure_datasets_exist()

    print("\n[INFO] Loading datasets...")
    data_fake = pd.read_csv(FAKE_CSV)
    data_true = pd.read_csv(TRUE_CSV)

    # Assign class labels
    data_fake['class'] = 0
    data_true['class'] = 1

    print(f"Initial Shapes - Fake: {data_fake.shape}, True: {data_true.shape}")

    # FIXED: Replaced brittle hardcoded index drops with position-based slicing (.iloc)
    # This prevents KeyError if the dataset has a different number of rows.
    print("[INFO] Creating manual testing sets (10 samples each)...")
    data_fake_manual_testing = data_fake.tail(10).copy()
    data_fake = data_fake.iloc[:-10]

    data_true_manual_testing = data_true.tail(10).copy()
    data_true = data_true.iloc[:-10]

    # Combine manual testing sets and save to a separate CSV for testing
    manual_testing = pd.concat([data_fake_manual_testing, data_true_manual_testing], axis=0)
    manual_testing.to_csv('manual_testing_samples.csv', index=False)
    print("[SUCCESS] Saved 20 manual testing samples to 'manual_testing_samples.csv'.")

    # Combine training sets
    data_merge = pd.concat([data_fake, data_true], axis=0)
    
    # Drop columns not required for model training
    data = data_merge.drop(['title', 'subject', 'date'], axis=1, errors='ignore')

    # Shuffle the combined dataset
    print("[INFO] Shuffling and cleaning data...")
    data = data.sample(frac=1).reset_index(drop=True)

    # Apply text cleaning
    data['text'] = data['text'].apply(wordopt)

    # Separate feature and target variables
    x = data['text']
    y = data['class']

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Feature extraction via TF-IDF Vectorization
    print("[INFO] Transforming text into TF-IDF features...")
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Dictionary to store accuracy scores
    scores = {}

    # 1. Logistic Regression
    print("[INFO] Training Logistic Regression...")
    LR = LogisticRegression(max_iter=1000)
    LR.fit(xv_train, y_train)
    pred_lr = LR.predict(xv_test)
    scores['Logistic Regression'] = accuracy_score(y_test, pred_lr)
    print(classification_report(y_test, pred_lr))

    # 2. Decision Tree Classifier
    print("[INFO] Training Decision Tree Classifier...")
    DT = DecisionTreeClassifier(random_state=42)
    DT.fit(xv_train, y_train)
    pred_dt = DT.predict(xv_test)
    scores['Decision Tree'] = accuracy_score(y_test, pred_dt)
    print(classification_report(y_test, pred_dt))

    # 3. Gradient Boosting Classifier
    print("[INFO] Training Gradient Boosting Classifier...")
    # Using small estimators for quick training on synthetic/sample datasets if needed
    GB = GradientBoostingClassifier(random_state=42, n_estimators=50 if is_synthetic else 100)
    GB.fit(xv_train, y_train)
    pred_gb = GB.predict(xv_test)
    scores['Gradient Boosting'] = accuracy_score(y_test, pred_gb)
    print(classification_report(y_test, pred_gb))

    # 4. Random Forest Classifier
    print("[INFO] Training Random Forest Classifier...")
    RF = RandomForestClassifier(random_state=42, n_estimators=50 if is_synthetic else 100)
    RF.fit(xv_train, y_train)
    pred_rf = RF.predict(xv_test)
    scores['Random Forest'] = accuracy_score(y_test, pred_rf)
    print(classification_report(y_test, pred_rf))

    print("\n[SUMMARY] Model Training Summary (Accuracy):")
    for model_name, score in scores.items():
        print(f" - {model_name}: {score * 100:.2f}%")

    # Serialize and Save Models and Vectorizer for future use
    print("\n[SAVE] Serializing and saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorization, f)
    with open(LR_MODEL_PATH, 'wb') as f:
        pickle.dump(LR, f)
    with open(DT_MODEL_PATH, 'wb') as f:
        pickle.dump(DT, f)
    with open(GB_MODEL_PATH, 'wb') as f:
        pickle.dump(GB, f)
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(RF, f)
    print("[SUCCESS] All models successfully trained and serialized to the 'models/' directory!")

def load_models_and_predict(news_text):
    """
    Loads saved models and predicts the class of a single news article.
    """
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(LR_MODEL_PATH)):
        print("[ERROR] Serialized models not found. Please train the models first!")
        return None

    # Load vectorizer and models
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(LR_MODEL_PATH, 'rb') as f:
        LR = pickle.load(f)
    with open(DT_MODEL_PATH, 'rb') as f:
        DT = pickle.load(f)
    with open(GB_MODEL_PATH, 'rb') as f:
        GB = pickle.load(f)
    with open(RF_MODEL_PATH, 'rb') as f:
        RF = pickle.load(f)

    # Preprocess input news text
    cleaned_text = wordopt(news_text)
    xv_text = vectorizer.transform([cleaned_text])

    # Class labels
    labels = {0: "Fake News (RED)", 1: "True/Genuine News (GREEN)"}

    predictions = {
        'Logistic Regression': labels[LR.predict(xv_text)[0]],
        'Decision Tree': labels[DT.predict(xv_text)[0]],
        'Gradient Boosting': labels[GB.predict(xv_text)[0]],
        'Random Forest': labels[RF.predict(xv_text)[0]]
    }
    
    return predictions

if __name__ == '__main__':
    print("=========================================")
    print("    FAKE NEWS DETECTOR ENGINE (FIXED)    ")
    print("=========================================")
    
    # Train if models are not present or if run directly
    if not os.path.exists(LR_MODEL_PATH):
        print("\n[INFO] Serialized models not detected. Starting training pipeline...")
        train_and_save_models()
    else:
        print("\n[INFO] Serialized models found in 'models/'. Ready for predictions.")

    # Interactive loop
    while True:
        print("\n-----------------------------------------")
        print("Options:")
        print("1. Enter a news article to check")
        print("2. Force retrain models")
        print("3. Exit")
        choice = input("Select an option (1-3): ").strip()

        if choice == '1':
            print("\nPaste the news article text below:")
            news_input = input(">> ").strip()
            if not news_input:
                print("[WARNING] News text cannot be empty.")
                continue

            print("\n[INFO] Analyzing news article...")
            preds = load_models_and_predict(news_input)
            if preds:
                print("\n[RESULTS] Prediction Results:")
                for clf_name, outcome in preds.items():
                    print(f" - {clf_name:22}: {outcome}")

        elif choice == '2':
            print("\n[INFO] Starting manual model retraining...")
            train_and_save_models()

        elif choice == '3':
            print("\n[INFO] Exiting Fake News Detector. Goodbye!")
            break
        else:
            print("[WARNING] Invalid choice. Please select 1, 2, or 3.")
