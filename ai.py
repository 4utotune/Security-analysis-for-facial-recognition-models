import numpy as np
import tensorflow as tf

# Carica i dati
x_train = np.load('x_dataset.npy')
y_train = np.load('y_dataset.npy')

# Calcola la dimensione del training set
print(len(x_train))
train_size = int(0.9 * len(x_train))
print(train_size)

# Suddividi i dati in un training set e un testing set
x_train, x_test = x_train[:train_size], x_train[train_size:]
y_train, y_test = y_train[:train_size], y_train[train_size:]
print(len(x_train))
print(len(x_test))
print(x_train.shape)
print(x_test.shape)

# Costruisci il modello
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(512,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(160 * 160 * 3, activation='sigmoid'),
    tf.keras.layers.Reshape((160, 160, 3))
])

# Compila il modello
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Addestra il modello
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.0)

# Valuta il modello sul testing set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
