import string
import email
# NLTK Library Downloaded from nltk.org
import nltk

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

# Combine the different parts of the email into a flat list of strings
def flatten_to_string(parts):
    ret = []
    # Checks to see if paarts is a string. If so, add it to the return string
    if type(parts) == str:
        ret.append(parts)
    # Checks to see if parts is a list. If so, parse through everything
    # recursively and add it to the return
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    # Checks to see if parts is just text. If so, return the entire thing?
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret

# Extract subject and body text from a single email file
def extract_email_test(path):
    