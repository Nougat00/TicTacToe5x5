"""
Prediction of CIFAR-10 dataset with 10 classes.

Authors: Jakub Gola & Bartosz Laskowski
"""

from keras import layers, models
from keras.datasets import cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

# Build a neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

# Configure the model for training
opt = Adam(learning_rate=0.02)
model.compile(opt,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the training data
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model on the test data
loss, acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {acc}, Test Loss: {loss}')

# Make predictions using the trained model
predictions = model.predict(test_images)
