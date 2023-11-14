import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Reads in the CSV into a Data Frame and then samples it
df = pd.read_csv('datasets/payment_fraud.csv')

# Replaces the non-numerical field paymentMethod into a binary feature
df = pd.get_dummies(df, columns=['paymentMethod'])

# Splits the data set into training and testing data sets.
# 2/3 is used for training and 1/3 is used for testing purposes
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'], test_size=0.33, random_state=17
)

# Create a LogisticRegression object to create a trained classifier model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make the predictions on the test set and then compare it to the actual set (ground truth labels)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))

