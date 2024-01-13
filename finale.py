import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ShowImagesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Scegli casualmente un'immagine dal set di test
        idx = np.random.randint(len(x_test))
        test_image = x_train[idx]
        test_label = y_train[idx]
        
        # Applica il modello all'immagine scelta
        decoded_image = model.predict(test_image[np.newaxis, ...])[0]
        
        # Visualizza l'immagine decodificata e la label
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(decoded_image)
        ax[0].set_title('Decoded Image')
        ax[1].imshow(test_label)
        ax[1].set_title('Label')
        plt.show() #bloccante


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

#prova

# Costruisci il modello

model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(512,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(160 * 160 * 3, activation='sigmoid'),
    tf.keras.layers.Reshape((160, 160, 3))
])

model2 = tf.keras.Sequential([
    tf.keras.layers.Reshape((16, 32, 1), input_shape=(512,)), 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(160 * 160 * 3, activation='sigmoid'),
    tf.keras.layers.Reshape((160, 160, 3))
])

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((16, 32, 1), input_shape=(512,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
        
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
        
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
        
    tf.keras.layers.Dense(160 * 160 * 3, activation='sigmoid'),
    tf.keras.layers.Reshape((160, 160, 3))
])

# Compila il modello
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

# Addestra il modello
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.0, callbacks=[ShowImagesCallback()])

# Valuta il modello sul testing set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


