import os
import pickle
import EmailParser
from IPython.display import HTML, display


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
            stems = EmailParser.load(path)
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

fp = 0
tp = 0
fn = 0
tn = 0

for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = EmailParser.load(path)
        if not stems:
            continue
        stems_set = set(stems)
        if stems_set & blacklist:
            if label == 1:
                fp = fp + 1
            else:
                tp = tp + 1
        else:
            if label == 1:
                tn = tn + 1
            else:
                fn = fn + 1


conf_matrix = [[tn, fp],
               [fn, tp]]
display(HTML('<table><tr>{}</tr></table>'.format(
    '</tr><tr>'.join('<td>{}</td>'.format(
        '</td><td>'.join(str(_) for _ in row))
                     for row in conf_matrix))))

count = tn + tp + fn + fp
percent_matrix = [["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)],
                  ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]]
display(HTML('<table><tr>{}</tr></table>'.format(
    '</tr><tr>'.join('<td>{}</td>'.format(
        '</td><td>'.join(str(_) for _ in row))
                     for row in percent_matrix))))

print("Classification accuracy: {}".format("{:.1%}".format((tp+tn)/count)))