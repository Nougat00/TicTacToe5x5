import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/data_banknote_authentication.csv')

X = data[['Variance', 'Skewness', 'Curtosis', 'Entropy']]
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

opt = Adam(learning_rate=0.02)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=20, batch_size=16)

loss, acc = model.evaluate(X_test, Y_test)
print(f'Loss (CC) on test data: {loss}')
print(f'Accuracy on test data: {acc}')

predictions = model.predict(X_test)

rounded_predictions = tf.round(predictions)

cm = confusion_matrix(Y_test, rounded_predictions)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()