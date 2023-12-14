"""
Prediction of banknote authenticity

Authors: Jakub Gola & Bartosz Laskowski
"""

import tensorflow as tf
import pandas as pd
from keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data = pd.read_csv('data/data_banknote_authentication.csv')

# Extracting features (X) and labels (Y)
X = data[['Variance', 'Skewness', 'Curtosis', 'Entropy']]
Y = data['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardizing the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating a simple neural network model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(100, activation='relu'),
    layers.Dense(1)
])

# Compiling the model with binary crossentropy loss and accuracy metric
opt = Adam(learning_rate=0.02)
model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

# Training the model on the training data
model.fit(X_train, Y_train, epochs=20, batch_size=16)

# Evaluating the model on the test data
loss, acc = model.evaluate(X_test, Y_test)
print(f'Loss (CC) on test data: {loss}')
print(f'Accuracy on test data: {acc}')

# Making predictions on the test data
predictions = model.predict(X_test)

# Rounding the predictions to obtain binary values
rounded_predictions = tf.round(predictions)

# Creating a confusion matrix and displaying it
cm = confusion_matrix(Y_test, rounded_predictions)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
