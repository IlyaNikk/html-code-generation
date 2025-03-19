# Modified version
__author__ = 'Taneem Jan, taneemishere.github.io'

import keras.src.callbacks
import datetime
from keras.layers import Input, MaxPooling2D, Conv2DTranspose, Reshape, Dense, MaxPooling2D, BatchNormalization, ReLU
from keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import *
from .Config import *
from .AModel import *


class autoencoder_image(AModel):
	def __init__(self, input_shape, output_size, output_path):
		AModel.__init__(self, input_shape, output_size, output_path)
		self.name = 'resnet50.weights'

		input_image = Input(shape=input_shape)

		base_model = ResNet50(weights=None, input_tensor=input_image, input_shape=input_shape, include_top=False)
		encoded = Dense(512, activation='relu')(base_model.output)
		encoded = MaxPooling2D()(encoded)

		self.encoder = Model(input_image, encoded, name='ResNet50_Encoder')

		x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(encoded)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)

		decoder = Model(encoded, x, name='Decoder')

		self.model = Model(input_image, decoder(encoded), name='Autoencoder')
		self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
		print('Model Summary info')
		self.model.summary()
		keras.utils.plot_model(
			self.model, to_file='full_model_resnet.png', show_shapes=True, show_layer_names=True,
			rankdir='TB', expand_nested=True, dpi=96
		)

	def fit_generator(self, generator, steps_per_epoch, callbacks):
		checkpoint_file_name = "{}/resnet50.weights.h5".format(self.output_path)
		self.model.load_weights(checkpoint_file_name)

		file_name = "resnet50_checkpoint_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".weights"
		checkpoint_filepath = "{}/{}.h5".format(self.output_path, file_name)
		model_checkpoint_callback = keras.src.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_loss',
			verbose=1,
			save_freq='epoch'
		)
		self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1, callbacks=[model_checkpoint_callback, *callbacks])
		self.save()
		self.encoder.save('resnet50_encoder.weights.h5')

	def predict_hidden(self, images):
		keras.utils.plot_model(
			self.model, to_file='full_model_resnet.png', show_shapes=True, show_layer_names=True,
			rankdir='TB', expand_nested=True, dpi=96
		)
		hidden_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer('max_pooling2d').output)
		keras.utils.plot_model(
			hidden_layer_model, to_file='hidden_model_resnet.png', show_shapes=True, show_layer_names=True,
			rankdir='TB', expand_nested=True, dpi=96
		)
		return hidden_layer_model.predict(images.reshape(-1, 256, 256, 3))
