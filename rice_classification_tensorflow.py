import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''Wczytywanie danych z pliku csv'''
data = pd.read_csv('data/riceClassification.csv')

'''Podział danych na cechy (X) i etykiety (Y)'''
X = data[['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Perimeter', 'Roundness', 'AspectRation']]
Y = data['Class']

'''Podział danych na zestawy treningowe i testowe'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

'''Standaryzacja danych'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''Definicja modelu TensorFlow'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

big_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(75, activation='gelu'),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

'''Kompilacja modelu'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
big_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''Trenowanie modelu'''
model.fit(X_train, Y_train, epochs=20, batch_size=16)
big_model.fit(X_train, Y_train, epochs=20, batch_size=16)


'''Ocena modelu na zestawie testowym'''
loss, acc = model.evaluate(X_test, Y_test)
loss_big, acc_big = big_model.evaluate(X_test, Y_test)
print(f'Loss (CC) on test data: {loss}')
print(f'Accuracy on test data: {acc}')
print(f'Loss (CC) on test data: {loss_big}')
print(f'Accuracy on test data: {acc_big}')

'''Przewidywanie na podstawie modelu'''
predictions = model.predict(X_test)
predictions_big = big_model.predict(X_test)