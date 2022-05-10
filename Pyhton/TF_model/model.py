import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class SequentialModel:
    def digits_model(self, train_images, train_labels, fit_epochs:int):
        # Define the model architecture.
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28)),
                keras.layers.Reshape(target_shape=(28, 28, 1)),
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Flatten(),
                keras.layers.Dense(10)
            ]
        )

        # Define how to train the model.
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        # Train the digit classification model.
        model.fit(train_images, train_labels, epochs=fit_epochs)
        model.summary()
        return model
    
    def iris_model(self, X, y):
        model = Sequential()

        model.add(Dense(64, activation='relu', input_shape=[4])) # shape: 4 -> 0,1,2,3,4 columns
        model.add(Dense(64))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(X, y, epochs=200)

        model.summary()
        return model