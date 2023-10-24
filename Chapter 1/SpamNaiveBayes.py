import os
# import email_read_util
import email

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

## The filenames used to link the data
DATA_DIR = 'datasets/trec07p/data/'
LABELS_FILE = 'datasets/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

## The list of labels
labels = {}

## Reads in the email files that are given
def read_email_files():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i + 1)
        email_str = email.extract_email_text(os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y

# Read the labels
with open(LABELS_FILE) as f:
    ## Parses through each line
    for line in f:
        ## Strips all leading and trailing whitespace characters
        line = line.strip()
        ## Tokenizes the line into substrings
        label, key = line.split()
        if label.lower() == 'ham':
            labels[key.split('/')[-1]] = 1
        else:
            labels[key.split('/')[-1]] = 0
        ## More compact way from the text book:
        # labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

## Prepares the raw data
X, y = read_email_files()
X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)),
    train_size=TRAINING_SET_RATIO, random_state = 2)

## Converts each email into a vector representation that MultinomialNB takes
## as an input
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Initialize the classifier and make label predictions
mnb = MultinomialNB()
mnb.fit(X_train_vector, y_train)
y_pred = mnb.predict(X_test_vector)
# Print results
print('Accuracy {:.3f}'.format(accuracy_score(y_test, y_pred)))