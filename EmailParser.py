## Most likely default libraries in Python
import string
import email
## NLTK Library Downloaded from nltk.org
import nltk

## List of different types of punctuation
punctuations = list(string.punctuation)
## List of the stop words in English
stopwords = set(nltk.corpus.stopwords.words('english'))
## ?
stemmer = nltk.PorterStemmer()

# Combine the different parts of the email into a flat list of strings
def flatten_to_string(parts):
    ret = []
    ## Checks to see if paarts is a string. If so, add it to the return string
    if type(parts) == str:
        ret.append(parts)
    ## Checks to see if parts is a list. If so, parse through everything
    ## recursively and add it to the return
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    ## Checks to see if parts is just text. If so, return the entire thing?
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret

# Extract subject and body text from a single email file
def extract_email_test(path):
    # Load a single email from an input file
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    # #Checks to see if the message exists or not
    if not msg:
        return ""
    
    # Read the email subject
    subject = msg['Subject']
    ## Checks to see if the subject is in the email
    if not subject:
        subject = ""
    
    # Read the email body
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)

    ## Checks to see if the body is in the email
    if not body:
        body = ""
    
    ## Return the subject and the body as a string representation
    return subject + ' ' + body

# Process a single email file into stemmed tokens
def load(path):
    ## Takes the email and turns it into a string to be parsed
    email_text = extract_email_test(path)
    ## Checks to see if the string contains anything
    if not email_text:
        return []
    
    # Tokenize the message
    tokens = nltk.word_tokenize(email_text)

    # Remove punctuation from tokens
    ## Parses through each of the tokens and gets rid of any punctuation
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]

    # Remove stopwords and stem tokens
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]
    return []
