import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

with open("training_data.dat", 'rb') as training_file:
    X_train_prepared, y_train = pickle.load(training_file)

print("Beginning processed")
classifier = SGDClassifier(random_state=42, max_iter=1e8, tol=1e-10)
classifier.fit(X_train_prepared, y_train)
print("Training finished")

with open("model.dat", 'wb') as file:
    pickle.dump(classifier, file)

print("Model saved")

with open("testing_data.dat", 'rb') as testing_file:
    X_test_prepared, y_test = pickle.load(testing_file)

y_pred = classifier.predict(X_test_prepared)
print('Percentage correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))

label_names = ['Square', 'Squiggle', 'Star', 'Triangle', 'None']
cmx = confusion_matrix(y_test, y_pred, labels=label_names)
print(cmx / cmx.sum(axis=1, keepdims=True))
