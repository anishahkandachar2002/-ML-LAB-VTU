import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Read the dataset
msg = pd.read_csv('document.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

print(msg.tail())

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(msg['message'], msg['labelnum'])

# Vectorize the text data
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Train the classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)

# Make predictions on the test set
pred = clf.predict(Xtest_dm)
print('prediction made are')
# Print the predictions
for doc, p in zip(Xtest, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))

# Evaluate the classifier
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))
