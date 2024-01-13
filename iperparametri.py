import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, ReLU, Input, Reshape
from tensorflow.keras.models import Model
import keras_tuner

class ShowImagesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Scegli casualmente un'immagine dal set di test
        idx = np.random.randint(len(x_test))
        test_image = x_test[idx]
        test_label = y_test[idx]
        
        # Applica il modello all'immagine scelta
        decoded_image = model.predict(test_image[np.newaxis, ...])[0]
        
        # Visualizza l'immagine decodificata e la label
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(decoded_image)
        ax[0].set_title('Decoded Image')
        ax[1].imshow(test_label)
        ax[1].set_title('Label')
        plt.show()


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

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(Input(shape=(512,)))
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(160 * 160 * 3, activation='sigmoid'))
    model.add(Reshape((160, 160, 3)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

model = build_model(keras_tuner.HyperParameters())

#Genero un insieme di modelli con iperparametri casuali e scelgo il modello con la minima perdita (loss) come modello migliore
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="loss",
    max_trials=2, #numero di volte di esecuzione dell'algoritmo di ricerca
    executions_per_trial=2, #numero di esecuzioni per ogni modello
    overwrite=True,
)

#tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=10, batch_size=32, validation_split=0.0)
# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build()
best_model.summary()

# Addestra il modello e verifica i risultati
best_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.0, callbacks=[ShowImagesCallback()])
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)