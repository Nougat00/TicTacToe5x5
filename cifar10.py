from keras import layers, models
from keras.datasets import cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

opt = Adam(learning_rate=0.02)
model.compile(opt,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

loss, acc = model.evaluate(test_images, test_labels)
print(acc, loss)

predictions = model.predict(test_images)