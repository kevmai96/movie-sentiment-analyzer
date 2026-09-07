import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import re, warnings

warnings.filterwarnings('ignore')

reviews_data = [
    # positive
    ("This movie was absolutely brilliant. The acting was superb and the story kept me on the edge of my seat.", 1),
    ("One of the best films I have ever seen. Outstanding performances from the entire cast.", 1),
    ("A masterpiece of modern cinema. The director has created something truly special.", 1),
    ("Incredibly moving and beautifully shot. I cried at the ending. Highly recommend.", 1),
    ("Fantastic film with amazing special effects and a gripping storyline.", 1),
    ("The performances were extraordinary. This film deserves every award it gets.", 1),
    ("A wonderful movie that kept me entertained throughout. Will definitely watch again.", 1),
    ("Stunning visuals combined with an emotional story. This is cinema at its finest.", 1),
    ("Excellent script, brilliant direction, and perfect casting. A must watch film.", 1),
    ("This film exceeded all my expectations. Genuinely one of the best of the year.", 1),

    # negative
    ("Terrible film. Boring plot with cardboard characters and awful dialogue.", 0),
    ("Worst movie I have seen in years. Complete waste of two hours of my life.", 0),
    ("Dreadful screenplay with zero character development. Avoid at all costs.", 0),
    ("The acting was wooden and unconvincing. The story made no sense at all.", 0),
    ("A complete disappointment. Boring, predictable, and poorly executed.", 0),
    ("This film was painfully slow. Nothing interesting happened for two hours.", 0),
    ("Awful special effects and a nonsensical plot. Money wasted on this rubbish.", 0),
    ("The characters were unlikeable and the ending was deeply unsatisfying.", 0),
    ("A mess from start to finish. The director clearly had no idea what he was doing.", 0),
    ("Dull, lifeless, and forgettable. One of the worst films of the decade.", 0),
]

df = pd.DataFrame(reviews_data, columns=['review', 'label'])
print(f"Dataset ready: {len(df)} reviews")

pos_count = (df['label'] == 1).sum()
neg_count = (df['label'] == 0).sum()
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

print(f"\nTotal reviews: {len(df)}")
print(f"Positive reviews: {pos_count}")
print(f"Negative reviews: {neg_count}")
print(f"Average review length: {df['word_count'].mean():.0f} words")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['clean_review'] = df['review'].apply(clean_text)

stop_words = [
    'the','a','an','and','or','but','in','on','at','to','for','of','with','is','was','are','were','be','been',
    'have','has','had','do','does','did','will','would','could','should','this','that','it','its','i','me','my',
    'you','your','he','she','they','we','his','her','their','all','not','no','so','as','by','from','what','which',
    'who','how','very','just','more','also','than','up','out','about','into','if','there','when','can','one'
]

def top_words(texts, n=10):
    all_words = ' '.join(texts).split()
    filtered  = [w for w in all_words if w not in stop_words and len(w) > 2]

    return Counter(filtered).most_common(n)

pos_words = top_words(df[df['label']==1]['clean_review'])
neg_words = top_words(df[df['label']==0]['clean_review'])

print("\nTop words in POSITIVE reviews:", pos_words)
print("Top words in NEGATIVE reviews:", neg_words)

X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# Model 1: Naive Bayes + CountVectorizer
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
    ('classifier', MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
nb_pred = nb_pipeline.predict(X_test)
results['Naive Bayes'] = {'pipeline': nb_pipeline, 'accuracy': accuracy_score(y_test, nb_pred)}

# Model 2: Logistic Regression + TF-IDF
lr_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
results['Logistic Regression'] = {'pipeline': lr_pipeline, 'accuracy': accuracy_score(y_test, lr_pred)}

# Model 3: Linear SVM + TF-IDF
svm_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
    ('classifier', LinearSVC(random_state=42, max_iter=2000))
])
svm_pipeline.fit(X_train, y_train)
svm_pred = svm_pipeline.predict(X_test)
results['Linear SVM'] = {'pipeline': svm_pipeline, 'accuracy': accuracy_score(y_test, svm_pred)}

best_name = max(results, key=lambda x: results[x]['accuracy'])
best_pipeline = results[best_name]['pipeline']

print("\nModel accuracies:")
for name, result in results.items():
    print(f"  {name}: {result['accuracy']*100:.1f}%")

print(f"Best model: {best_name}")
print("\nType a movie review and press ENTER to analyse it (type 'quit' to exit):")

while True:
    user_review = input("Your review: ").strip()

    if user_review.lower() in ['quit', 'exit', 'q', '']:
        print("Exiting review analyser.")
        break
    if len(user_review.split()) < 3:
        print("Please write at least a few words.\n")
        continue

    clean = clean_text(user_review)

    nb_result  = nb_pipeline.predict([clean])[0]
    lr_result  = lr_pipeline.predict([clean])[0]
    svm_result = svm_pipeline.predict([clean])[0]

    votes = [nb_result, lr_result, svm_result]
    final = 1 if sum(votes) >= 2 else 0

    sentiment_label = "POSITIVE" if final == 1 else "NEGATIVE"

    print(f"\nNaive Bayes: {'Positive' if nb_result  == 1 else 'Negative'}")
    print(f"Logistic Regression: {'Positive' if lr_result  == 1 else 'Negative'}")
    print(f"Linear SVM: {'Positive' if svm_result == 1 else 'Negative'}")

    print(f"Final say: {sentiment_label}\n")
