"""
This script performs classification on rice data using a Decision Tree classifier
and visualizes the results through a scatter plot and a confusion matrix.

Output:
- Scatter plot of the input data with two classes: Jasminen (0) and Gonen (1).
- Accuracy metrics

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the rice data from the CSV file
input_file = 'data/riceClassification.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate the data into two classes: Jasminen (0) and Gonen (1)
Jasminen = np.array(X[y == 0])
Gonen = np.array(X[y == 1])

# Visualize the input data through a scatter plot
plt.figure()
plt.scatter(Jasminen[:, 0], Jasminen[:, 1], s=75, facecolors='black', linewidth=1, marker='x')
plt.scatter(Gonen[:, 0], Gonen[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Create and train the Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = classifier.predict(X_test)

class_names = ['Jasminen', 'Gonen']
# Display classification reports for both the training and test datasets
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

plt.show()
