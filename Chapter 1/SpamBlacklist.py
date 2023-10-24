import os
import pickle
import email_read_util

from nltk.corpus import words

DATA_DIR = 'datasets/trec07p/data'
LABELS_FILE = 'datasets/trec07p/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

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

# Split corpus into training and test sets
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist) * TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist) * TRAINING_SET_RATIO):]

## Parse through each filename in the training set
if not os.path.exists('blacklist.pkl'):
    for filename in X_train:
        path = os.path.join(DATA_DIR, filename)
        if filename in labels:
            label = labels[filename]
            stems = load(path)
            if not stems:
                continue
            elif label == 0:
                spam_words.update(stems)
            else:
                continue
        blacklist = spam_words - ham_words
        pickle.dump(blacklist, open('blacklist.pkl', 'wb'))
else:
    blacklist = pickle.load(open('blacklist.pkl', 'rb'))

## Prints out the total number of loaded tokens
print('Blacklist of {} tokens successfully built/loaded'.format(len(blacklist)))

word_set = set(words.words())
word_set.intersection(blacklist)