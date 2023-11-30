"""
This script performs classification on wheat data using a Support Vector Machine (SVM) classifier
with a radial basis function (RBF) kernel. It visualizes the results through a confusion matrix.

Output:
- Accuracy metrics

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the wheat data from the CSV file
input_file = 'data/wheatClassification.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Create and train the Support Vector Machine (SVM) classifier
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = svc.predict(X_test)

class_names = ['Class1', 'Class2', 'Class3']
# Display classification reports for both the training and test datasets
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, svc.predict(X_train), target_names=class_names, zero_division=0.0))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0.0))
print("#" * 40 + "\n")

# Show the plot
plt.show()
