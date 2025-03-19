import os
import tensorflow as tf

# Инициализируем сессию TensorFlow
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import sys
from classes.dataset.Generator import *
from classes.model.Main_Model import *
from classes.BatchTensorBoard import *
from classes.model.autoencoder_image import *
from keras.backend import clear_session

# Импорт стандартного TensorBoard callback
from tensorflow.keras.callbacks import TensorBoard

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

    # Генератор для изображений (для автоэнкодера)
    generator_images = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                                input_shape=input_shape, generate_binary_sequences=True,
                                                images_only=True)

    # Создаем директорию для логов TensorBoard, если её нет
    log_dir_autoencoder = os.path.join(output_path, "logs_autoencoder")
    if not os.path.exists(log_dir_autoencoder):
        os.makedirs(log_dir_autoencoder)

    log_dir_text = os.path.join(output_path, "logs_text")
    if not os.path.exists(log_dir_text):
        os.makedirs(log_dir_text)

    # Стандартный TensorBoard callback (логирование графа и гистограмм раз в эпоху)
    tensorboard_callback_autoencoder = TensorBoard(log_dir=log_dir_autoencoder, histogram_freq=1, write_graph=True)
    tensorboard_callback_autoencoder
    tensorboard_callback_lstm = TensorBoard(log_dir=log_dir_text, histogram_freq=1, write_graph=True)

    # Наш кастомный callback для логирования метрик после каждого батча
    batch_tensorboard_callback_autoencoder = BatchTensorBoard(log_dir=log_dir_autoencoder)
    batch_tensorboard_callback_lstm = BatchTensorBoard(log_dir=log_dir_text)

    # For training of autoencoders
    if train_autoencoder:
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.fit_generator(generator_images, steps_per_epoch=steps_per_epoch,
                                        callbacks=[tensorboard_callback_autoencoder, batch_tensorboard_callback_autoencoder])
        clear_session()

    # Training of our main-model
    model = Main_Model(input_shape, output_size, output_path)
    model.fit_generator(generator,
                        validation_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=[tensorboard_callback_lstm, batch_tensorboard_callback_lstm])

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
