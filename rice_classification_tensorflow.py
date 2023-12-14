"""
Prediction of rice choice with two classes: Jasminen (0) and Gonen (1).

Authors: Jakub Gola & Bartosz Laskowski
"""

import tensorflow as tf
import pandas as pd
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file
data = pd.read_csv('data/riceClassification.csv')

# Extract features (X) and target variable (Y) from the dataset
X = data[['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Perimeter', 'Roundness', 'AspectRation']]
Y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Create a more complex neural network model
big_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(75, activation='gelu'),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

# Define optimizers and compile the models
opt = Adam(learning_rate=0.02)
opt1 = Adam(learning_rate=0.02)
model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
big_model.compile(opt1, loss='binary_crossentropy', metrics=['accuracy'])

# Train the models on the training data
model.fit(X_train, Y_train, epochs=20, batch_size=16)
big_model.fit(X_train, Y_train, epochs=50, batch_size=16)

# Evaluate the models on the test data
loss, acc = model.evaluate(X_test, Y_test)
loss_big, acc_big = big_model.evaluate(X_test, Y_test)

# Print the evaluation results
print(f'Loss (CC) on test data: {loss}')
print(f'Accuracy on test data: {acc}')
print(f'Loss (CC) on test data: {loss_big}')
print(f'Accuracy on test data: {acc_big}')

# Make predictions using the trained models
predictions = model.predict(X_test)
predictions_big = big_model.predict(X_test)
