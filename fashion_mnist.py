import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.losses import SparseCategoricalCrossentropy

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

loss, acc = model.evaluate(test_images, test_labels)

print(acc, loss)

predictions = model.predict(test_images)