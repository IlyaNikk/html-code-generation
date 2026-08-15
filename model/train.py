"""Train the main model (and optionally the autoencoder).

Two invocation styles supported:

  Profile-based (preferred):
      python3 model/train.py --profile web
      python3 model/train.py --profile web_generated --train-autoencoder

  Legacy positional (backward-compat with old Makefile / docs):
      python3 model/train.py <training_set> <eval_set> <output_path> [autoencoder_flag]

The profile name is persisted to `<output_path>/training_profile.txt` so every
downstream script (predict_one.py, functional-test.py, the in-training
TestingCallback, …) automatically picks up the right DSL mapping and eval
set via `classes.dataset.profiles.load(weights_path)` — no per-file edits
when you switch datasets.
"""

import argparse
import json
import os
import tensorflow as tf

# Инициализируем сессию TensorFlow
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import sys
from classes.dataset.Generator import *
from classes.dataset import profiles as dataset_profiles
from classes.model.Main_Model import *
from classes.BatchTensorBoard import *
from classes.model.autoencoder_image import *
from keras.backend import clear_session

# Импорт стандартного TensorBoard callback
from tensorflow.keras.callbacks import TensorBoard


def run(input_path, input_validation_path, output_path, profile_name, train_autoencoder=False,
        train_steps_fraction=1.0, random_seed=1234):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)
    # Запоминаем выбранный профиль, чтобы инференс-скрипты сами подхватили
    # правильный DSL mapping и eval_set из sidecar-файла.
    dataset_profiles.save(output_path, profile_name)
    with open(os.path.join(output_path, "training_run_config.json"), "w") as destination:
        json.dump({
            "profile": profile_name,
            "training_set": input_path,
            "eval_set": input_validation_path,
            "train_autoencoder": train_autoencoder,
            "train_steps_fraction": train_steps_fraction,
            "seed": random_seed,
        }, destination, indent=2)
        destination.write("\n")

    gui_paths, img_paths = Dataset.load_paths_only(input_path)

    input_shape = dataset.input_shape
    output_size = dataset.output_size
    full_steps_per_epoch = max(1, int(dataset.size / BATCH_SIZE))
    if train_steps_fraction <= 0 or train_steps_fraction > 1:
        raise SystemExit("train_steps_fraction must be in (0, 1], got {}".format(train_steps_fraction))
    steps_per_epoch = max(1, int(full_steps_per_epoch * train_steps_fraction))
    print("[train] steps_per_epoch={} (fraction={} from full {})".format(
        steps_per_epoch, train_steps_fraction, full_steps_per_epoch), flush=True)

    voc = Vocabulary()
    voc.retrieve(output_path)

    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, input_shape=input_shape,
                                         generate_binary_sequences=True)

    validation_dataset = Dataset()
    validation_dataset.load(input_validation_path, generate_binary_sequences=True)

    gui_validation_paths, img_validation_paths = Dataset.load_paths_only(input_validation_path)
    input_validation_shape = validation_dataset.input_shape

    # Cap validation batches per epoch. validation_dataset.size counts SLIDING WINDOWS
    # (~30 per .gui), so 250 eval files balloon to ~117 batches at batch_size=64 — на CPU
    # это 5–10 минут «тихой» валидации между концом тренировочной фазы и началом
    # on_epoch_end callbacks, из-за чего TestingCallback не успевает напечатать ничего
    # видимого. Реальное качество мерим в TestingCallback (BLEU/diff на полном eval-сете),
    # а val_loss оставляем как быстрый шумный сигнал по ~20 батчам.
    VAL_STEPS_CAP = 50
    full_val_steps = max(1, int(validation_dataset.size / BATCH_SIZE))
    validation_steps = min(VAL_STEPS_CAP, full_val_steps)
    print("[train] validation_steps={} (capped from {})".format(validation_steps, full_val_steps), flush=True)

    validation_generator = Generator.data_generator(
        voc,
        gui_validation_paths,
        img_validation_paths,
        batch_size=BATCH_SIZE,
        input_shape=input_validation_shape,
        generate_binary_sequences=True
    )

    # (opt #1) Отдельный генератор для автоэнкодера — итерируется по уникальным
    # изображениям, без раздувания DSL sliding-window'ом (раньше один и тот же
    # img попадал в батч ~30 раз, гробя batch-эффективную численность).
    generator_images = Generator.image_only_generator(img_paths, batch_size=BATCH_SIZE)
    ae_steps_per_epoch = max(1, len(img_paths) // BATCH_SIZE)

    # Создаем директорию для логов TensorBoard, если её нет
    log_dir_autoencoder = os.path.join(output_path, "logs_autoencoder")
    if not os.path.exists(log_dir_autoencoder):
        os.makedirs(log_dir_autoencoder)

    log_dir_text = os.path.join(output_path, "logs_text")
    if not os.path.exists(log_dir_text):
        os.makedirs(log_dir_text)

    # Стандартный TensorBoard callback (логирование графа и гистограмм раз в эпоху)
    tensorboard_callback_autoencoder = TensorBoard(log_dir=log_dir_autoencoder, histogram_freq=1, write_graph=True)
    tensorboard_callback_lstm = TensorBoard(log_dir=log_dir_text, histogram_freq=1, write_graph=True)

    # Наш кастомный callback для логирования метрик после каждого батча
    batch_tensorboard_callback_autoencoder = BatchTensorBoard(log_dir=log_dir_autoencoder)
    batch_tensorboard_callback_lstm = BatchTensorBoard(log_dir=log_dir_text)

    # For training of autoencoders
    if train_autoencoder:
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        # ae_steps_per_epoch использует len(img_paths), а не dataset.size (sliding windows).
        autoencoder_model.fit_generator(generator_images, steps_per_epoch=ae_steps_per_epoch,
                                        callbacks=[tensorboard_callback_autoencoder, batch_tensorboard_callback_autoencoder])
        clear_session()

    # Training of our main-model.
    # Передаём фактический total_training_steps в Main_Model, чтобы CosineDecay-расписание
    # масштабировалось под реальный размер датасета и Config.EPOCHS, а не клампилось на 1e-8
    # к середине обучения (см. fix D1).
    model = Main_Model(
        input_shape, output_size, output_path,
        total_training_steps=steps_per_epoch * EPOCHS,
    )
    model.fit_generator(generator,
                        validation_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=[tensorboard_callback_lstm, batch_tensorboard_callback_lstm])


def _parse_args(argv):
    """Parse CLI args. Supports both the new --profile form and the legacy positional form."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None,
                        help="dataset profile name (see classes/dataset/profiles.py). Default: web.")
    parser.add_argument("--train-autoencoder", action="store_true",
                        help="train the autoencoder before the main model")
    parser.add_argument("--steps-fraction", type=float, default=None,
                        help="fraction of training batches per epoch; defaults to the selected profile setting")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed for NumPy and TensorFlow; stored with the trained checkpoint")
    parser.add_argument("paths", nargs="*",
                        help="(legacy) <training_set> <eval_set> <output_path> [autoencoder_flag]")
    args = parser.parse_args(argv)

    if len(args.paths) >= 3:
        # Legacy positional form — keep working so old Makefile invocations / docs don't break.
        input_path  = args.paths[0]
        val_path    = args.paths[1]
        output_path = args.paths[2]
        train_autoencoder = (len(args.paths) >= 4 and str(args.paths[3]) == "1") or args.train_autoencoder
        train_steps_fraction = args.steps_fraction if args.steps_fraction is not None else 1.0
        # We don't know which profile this corresponds to — best-effort match by training_set path,
        # else fall back to "web". The sidecar's main purpose is to tell inference scripts which
        # DSL mapping to use; mis-tagging only matters if the legacy caller is using a non-default set.
        profile_name = "web"
        for name, prof in dataset_profiles.PROFILES.items():
            if os.path.normpath(prof["training_set"]) == os.path.normpath(input_path):
                profile_name = name
                break
    else:
        # New form — resolve everything from the named profile.
        profile_name = args.profile or "web"
        profile = dataset_profiles.get(profile_name)
        input_path  = profile["training_set"]
        val_path    = profile["eval_set"]
        output_path = profile["output_dir"]
        train_autoencoder = args.train_autoencoder
        train_steps_fraction = (
            args.steps_fraction
            if args.steps_fraction is not None
            else profile.get("train_steps_fraction", 1.0)
        )

    return input_path, val_path, output_path, profile_name, train_autoencoder, train_steps_fraction, args.seed


if __name__ == "__main__":
    input_path, input_validation_path, output_path, profile_name, train_autoencoder, train_steps_fraction, seed = _parse_args(sys.argv[1:])
    print("Training with profile={!r}: input={} val={} output={} train_autoencoder={} steps_fraction={} seed={}".format(
        profile_name, input_path, input_validation_path, output_path, train_autoencoder, train_steps_fraction, seed))
    run(input_path, input_validation_path, output_path, profile_name, train_autoencoder=train_autoencoder,
        train_steps_fraction=train_steps_fraction, random_seed=seed)
