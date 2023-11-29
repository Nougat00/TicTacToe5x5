"""
This script performs classification on rice data using a Support Vector Machine (SVM) classifier
with a radial basis function (RBF) kernel. It visualizes the results through a confusion matrix.

Output:
- Confusion matrix with a visualization.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the rice data from the CSV file
input_file = 'data/riceClassification.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Create and train the Support Vector Machine (SVM) classifier
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = svc.predict(X_test)

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()
print(cm)

# Show the plot
plt.show()
