# Modified version
__author__ = 'Taneem Jan, taneemishere.github.io'

import keras.src.callbacks
from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Flatten
from keras.models import Sequential, Model
import tensorflow as tf
from keras import *
from .Config import *
from .AModel import *
from .autoencoder_image import *
import matplotlib.pyplot as plt


class Main_Model(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "Main_Model.weights"

        visual_input = Input(shape=input_shape)

        # Load the pre-trained autoencoder model
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.load('autoencoder.weights')
        autoencoder_model.model.load_weights('../bin/autoencoder.weights.h5')

        hidden_layer_model_freeze = Model(inputs=autoencoder_model.model.input,
                                          outputs=autoencoder_model.model.get_layer('encoded_layer').output)
        hidden_layer_input = hidden_layer_model_freeze(visual_input)

        # Additional layers before concatenation
        hidden_layer_model = Flatten()(hidden_layer_input)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_result = RepeatVector(CONTEXT_LENGTH)(hidden_layer_model)

        # Make sure the loaded hidden_layer_model_freeze will no longer be updated
        for layer in hidden_layer_model_freeze.layers:
            layer.trainable = False

        # The same language model used by Tony Beltramelli
        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([hidden_layer_result, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    def fit_generator(self, generator, validation_generator, steps_per_epoch, validation_steps):
        self.model.summary()
        file_name = "autoencoder_main_model_checkpoint_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".weights"
        checkpoint_filepath = "{}/{}.h5".format(self.output_path, file_name)
        model_checkpoint_callback = keras.src.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1,
            save_freq='epoch'
        )

        checkpoint_file_name = "{}/{}.h5".format(self.output_path, self.name)
        self.model.load_weights(checkpoint_file_name)
        # loss, acc = self.model.evaluate(generator)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        print(steps_per_epoch)
        print(validation_steps)
        history = self.model.fit(
            generator,
            validation_data=validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[model_checkpoint_callback])
        self.save()
        self.visualizeReult(history)

    def predict(self, image, partial_caption):
        return self.model.predict((image, partial_caption), verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict((images, partial_captions), verbose=1)

    def visualizeReult(self, history):
        acc = history .history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
