# train_classifier_improved.py
import json
import joblib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV  # for probability estimates

def main():
    # 1. Load all datasets
    with open("Training Data/plant_diagnostics_final.json", "r", encoding="utf-8") as f:
        plant_data = json.load(f)
    with open("Training Data/farmbot_commands_final_p2.json", "r", encoding="utf-8") as f:
        farmbot_data = json.load(f)
    with open("Training Data/model_commands.json", "r", encoding="utf-8") as f:
        model_cmd_data = json.load(f)
    with open("Training Data/out_of_domain.json", "r", encoding="utf-8") as f:
        ood_data = json.load(f)

    # 2. Combine samples
    samples = []
    for entry in plant_data:
        if entry.get("type") == "plant_care":
            samples.append((entry["instruction"], "plant"))
    for entry in farmbot_data:
        if entry.get("type") == "farmbot_command":
            samples.append((entry["instruction"], "farmbot"))
    for entry in model_cmd_data:
        samples.append((entry["instruction"], "model_command"))
    for entry in ood_data:
        samples.append((entry["instruction"], "out_of_domain"))

    # Print class distribution
    from collections import Counter
    print("Class distribution:", Counter([l for _, l in samples]))

    # 3. Train/test split
    texts, labels = zip(*samples)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 4. Build pipeline
    # Use character n-grams in addition to word n-grams for better short-text handling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),       # unigrams, bigrams, trigrams
        max_features=10000,
        analyzer='char_wb',        # character n-grams (word-boundary aware)
        sublinear_tf=True
    )
    # LinearSVC is faster and often better for text; wrap for probability
    svc = LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
        dual=False
    )
    clf = CalibratedClassifierCV(svc, cv=3)  # enables predict_proba

    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf)
    ])

    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nTest accuracy: {accuracy:.3f}")

    # 5. Save model
    joblib.dump(pipeline, "query_classifier.pkl")
    print("Classifier saved as query_classifier.pkl")

    # 6. Detailed evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=pipeline.classes_))

if __name__ == "__main__":
    main()