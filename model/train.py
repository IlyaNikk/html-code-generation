from __future__ import print_function
from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com :: original author of the paper'
# for modified version
__author__ = 'Taneem Jan, taneemishere.github.io'

import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import sys
from classes.dataset.Generator import *
from classes.model.Main_Model import *
from classes.model.autoencoder_image import *
from keras.backend import clear_session


# removed some training parameters, to make the code input easier
def run(input_path, input_validation_path, output_path, train_autoencoder=False):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    gui_paths, img_paths = Dataset.load_paths_only(input_path)

    input_shape = dataset.input_shape
    output_size = dataset.output_size
    steps_per_epoch = int(dataset.size / BATCH_SIZE)

    voc = Vocabulary()
    voc.retrieve(output_path)

    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, input_shape=input_shape,
                                         generate_binary_sequences=True)

    validation_dataset = Dataset()
    validation_dataset.load(input_validation_path, generate_binary_sequences=True)

    gui_validation_paths, img_validation_paths = Dataset.load_paths_only(input_validation_path)
    input_validation_shape = validation_dataset.input_shape

    validation_steps = int(validation_dataset.size / BATCH_SIZE)

    validation_generator = Generator.data_generator(
        voc,
        gui_validation_paths,
        img_validation_paths,
        batch_size=BATCH_SIZE,
        input_shape=input_validation_shape,
        generate_binary_sequences=True
    )

    # Included a generator for images only as an input for autoencoders
    generator_images = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                                input_shape=input_shape, generate_binary_sequences=True,
                                                images_only=True)

    # For training of autoencoders
    if train_autoencoder:
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.fit_generator(generator_images, steps_per_epoch=steps_per_epoch)
        clear_session()

    # Training of our main-model
    model = Main_Model(input_shape, output_size, output_path)
    model.fit_generator(generator, validation_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <train_autoencoder default: 0>")
        exit(0)
    else:
        input_path = argv[0]
        input_validation_path = argv[1]
        output_path = argv[2]
        train_autoencoder = False if len(argv) < 4 else True if int(argv[3]) == 1 else False

    run(input_path, input_validation_path, output_path, train_autoencoder=train_autoencoder)
