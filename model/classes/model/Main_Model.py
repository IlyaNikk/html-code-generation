# Modified version
__author__ = 'Taneem Jan, taneemishere.github.io'

import keras.src.callbacks
from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Flatten, Embedding, Attention, TimeDistributed
from keras.models import Sequential, Model
import tensorflow as tf
from keras import *
from .Config import *
from .AModel import *
from .autoencoder_image import *
import matplotlib.pyplot as plt
from .Main_Model_Testing_Callback import *

class Main_Model(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "Main_Model.weights"

        # Load the pre-trained autoencoder model
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.load('resnet50.weights')
        autoencoder_model.model.load_weights('{}/resnet50.weights.h5'.format(output_path))

        visual_input = Input(shape=input_shape)

        hidden_layer_model_freeze = Model(inputs=autoencoder_model.model.input,
                                          outputs=autoencoder_model.model.get_layer('max_pooling2d').output)
        hidden_layer_input = hidden_layer_model_freeze(visual_input)
        #
        # Additional layers before concatenation
        hidden_layer_model = Flatten()(hidden_layer_input)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        # Получаем последовательность с фиксированной длиной CONTEXT_LENGTH
        hidden_layer_result = RepeatVector(CONTEXT_LENGTH)(hidden_layer_model)

        # Make sure the loaded hidden_layer_model_freeze will no longer be updated
        for layer in hidden_layer_model_freeze.layers:
            layer.trainable = False

        # Функция для трансформерного блока (encoder)
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
            x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Add()([x, inputs])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            x_ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            x_ff = tf.keras.layers.Dense(inputs.shape[-1])(x_ff)
            x_ff = tf.keras.layers.Dropout(dropout)(x_ff)
            x = tf.keras.layers.Add()([x, x_ff])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            return x

        # Вход для текстовой части (например, представление предыдущих слов)
        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        # Применяем трансформерный блок к текстовой последовательности
        encoded_text = transformer_encoder(textual_input, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

        # Объединяем визуальное и текстовое представления (конкатенация по последнему измерению)
        combined = concatenate([hidden_layer_result, encoded_text], axis=-1)
        # Применяем ещё один трансформерный блок к объединённой последовательности
        combined = transformer_encoder(combined, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)
        # Глобальное усреднение по временной оси
        pooled = tf.keras.layers.GlobalAveragePooling1D()(combined)
        # Финальный классификатор с softmax для предсказания следующего слова/токена
        output = Dense(output_size, activation='softmax')(pooled)

        self.model = Model(inputs=[visual_input, textual_input], outputs=output)

        # Задаем начальную скорость обучения и параметры косинусного расписания
        initial_learning_rate = 0.1
        decay_steps = 10000  # число шагов, за которое скорость обучения убывает до минимального значения
        alpha = 0.0001  # конечное значение скорости обучения будет равно initial_learning_rate * alpha
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

    def fit_generator(self, generator, validation_generator, steps_per_epoch, validation_steps, callbacks):
        self.model.summary()
        file_name = "seq2seq_checkpoint_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".weights"
        checkpoint_filepath = "{}/{}.h5".format(self.output_path, file_name)
        model_checkpoint_callback = keras.src.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1,
            save_freq='epoch'
        )

        # checkpoint_file_name = "{}/main_model_checkpoint_07_03_2025_02_23.weights.h5".format(self.output_path)
        # self.model.load_weights(checkpoint_file_name)
        # loss, acc = self.model.evaluate(generator)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        testing_callback = TestingCallback()
        keras.utils.plot_model(
            self.model, to_file='se2seq_resnet.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )

        history = self.model.fit(
            generator,
            validation_data=validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[model_checkpoint_callback, testing_callback, *callbacks])
        self.save()
        self.visualizeReult(history)

    def predict(self, image, partial_caption):
        return self.model.predict((image, partial_caption), verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict((images, partial_captions), verbose=1)

    def visualizeReult(self, history):
        acc = history.history['accuracy']
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
