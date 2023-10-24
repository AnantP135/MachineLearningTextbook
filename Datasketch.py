import os
import pickle
# import email_read_util
from datasketch import MinHash, MinHashLSH

DATA_DIR = 'datasets/trec07p/data'
LABELS_FILE = 'datasets/trec07p/index'
TRAINING_SET_RATIO = 0.7

labels = {}

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
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

# Extract only spam files for inserting into the LSH matcher
spam_files = [x for x in X_train if labels[x] == 0]

# Initialize MinHashLSH matcher with a Jaccard
# threshold of 0.5 and 128 MinHash permutation functions
lsh = MinHashLSH(threshhold = 0.5, num_perm = 128)

# Populate the LSH matcher with training spam MinHashes
for idx, f in enumerate(spam_files):
    ## Get a 128 Bit hash?
    minhash = MinHash(num_perm = 128)
    ## Loads the path using the function defined earlier
    stems = load(os.path.join(DATA_DIR, f))
    ## Checks to see if there is something in stems
    if len(stems) < 2: continue
    for s in stems:
        minhash.update(s.encode('utf-8'))
    lsh.insert(f, minhash)

def lsh_predict_label(stems):
    '''
    Queries the LSH matcher and returns:
        0 if predicted spam
        1 if predicted ham
        −1 if parsing error
    '''
    minhash = MinHash(num_perm = 128)
    if len(stems) < 2:
        return -1
    for s in stems:
        minhash.update(s.encode('utf-8'))
    matches = lsh.query(minhash)
    if matches:
        return 0
    else:
        return 1
