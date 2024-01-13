import numpy as np
from tensorflow.keras import layers, models

# Caricamento dei dati
x_train = np.load('x_dataset.npy')
y_train = np.load('y_dataset.npy')

# Suddivisione dei dati in training set e test set
"""
num_samples = x_train.shape[0]
num_train_samples = int(num_samples * 0.9)
num_test_samples = num_samples - num_train_samples

x_train = x_train[:num_train_samples]
y_train = y_train[:num_train_samples]
x_test = x_train[num_train_samples:]
y_test = y_train[num_train_samples:]
"""
# Calcola la dimensione del training set
print(len(x_train))
print(x_train[0].shape)
train_size = int(0.9 * len(x_train))

# Suddividi i dati in un training set e un testing set
x_train, x_test = x_train[:train_size], x_train[train_size:]
y_train, y_test = y_train[:train_size], y_train[train_size:]


# Creazione del modello
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(512,)),
    layers.Dense(3 * 3 * 128, activation='relu'),
    layers.Reshape((3, 3, 128)),
    layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
    layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
    layers.Conv2DTranspose(3, kernel_size=3, strides=1, activation='sigmoid', padding='same'),
])

model.summary()

# Compilazione del modello
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Addestramento del modello
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.0)

# Valutazione del modello sul test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: %.2f' % (loss*100))