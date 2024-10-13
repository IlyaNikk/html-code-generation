# Modified version
__author__ = 'Taneem Jan, taneemishere.github.io'

import keras.src.callbacks
import datetime
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape, Dense
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from keras import *
from .Config import *
from .AModel import *


class autoencoder_image(AModel):
	def __init__(self, input_shape, output_size, output_path):
		AModel.__init__(self, input_shape, output_size, output_path)
		self.name = 'autoencoder.weights'

		input_image = Input(shape=input_shape)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(input_image)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoded = Dropout(0.25, name='encoded_layer')(encoder)

		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(encoded)
		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(32, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(3, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoded = Dropout(0.25)(decoder)

		# decoder = Dense(256*256*3)(decoder)
		# decoded = Reshape(target_shape=input_shape)(decoder)

		self.model = Model(input_image, decoded)
		self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
		self.model.summary()

	def fit_generator(self, generator, steps_per_epoch):
		file_name = "autoencoder_checkpoint_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".weights"
		checkpoint_filepath = "{}/{}.h5".format(self.output_path, file_name)
		model_checkpoint_callback = keras.src.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_loss',
			verbose=1,
			save_freq='epoch'
		)
		self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=4, verbose=1, callbacks=[model_checkpoint_callback])
		self.save()

	def predict_hidden(self, images):
		hidden_layer_model = Model(inputs=self.input, outputs=self.get_layer('encoded_layer').output)
		return hidden_layer_model.predict(images)
