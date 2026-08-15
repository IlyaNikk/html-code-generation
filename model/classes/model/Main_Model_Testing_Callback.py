import os
import re
import sys
import time
import keras.src.callbacks

sys.path.append('./')

from ..Sampler import *
from ..dataset import profiles as dataset_profiles
from .Config import CONTEXT_LENGTH, IMAGE_SIZE
from compiler.classes.Compiler import *
from ..test_classes.Functional_Test import *
from ..test_classes.BLEU import *
# Намеренно ПОСЛЕ wildcard-импортов выше: иначе любой `from ... import *` из
# test_classes мог затереть `Utils` чем-нибудь не тем (в репо есть два класса
# с этим именем — model/Utils для картинок и compiler/Utils для рендеринга DSL).
# Эта строка делает финальное связывание `Utils` именно с нужным.
from ..Utils import Utils

class TestingCallback(keras.callbacks.Callback):
    def __init__(self, trained_weights_path, logs_path=None):
        super().__init__()
        self.trained_weights_path = trained_weights_path
        self.logs_path = logs_path or os.path.join(trained_weights_path, "testing_callback_logs.txt")

    def on_epoch_end(self, epoch, logs=None):
        meta_dataset = np.load("{}/meta_dataset.npy".format(self.trained_weights_path), allow_pickle=True)
        input_shape = meta_dataset[0]
        output_size = meta_dataset[1]

        # Resolve dataset-specific paths from the sidecar written by train.py.
        profile = dataset_profiles.load(self.trained_weights_path)
        dsl_path = profile["dsl_mapping"]
        input_path = profile["eval_set"]

        sampler = Sampler(self.trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)
        compiler = Compiler(dsl_path)

        functional_test_instance = FunctionalTest(self.model, sampler, compiler, input_path)

        # Берём первые 10 .gui-файлов из eval-сета активного профиля. Раньше тут
        # был хардкод 10 UUID из оригинального датасета, из-за чего после переключения
        # на синтетический сет callback падал с FileNotFoundError.
        all_gui = sorted(f for f in os.listdir(input_path) if re.search(r"\.gui$", f))
        gui_files = all_gui[:10]

        # Видимый прогресс: без flush=True stdout буферизуется и кажется, что обучение
        # «зависло» после каждой эпохи — на самом деле callback просто долго работает
        # (Sampler.predict_greedy делает до 150 model.predict со скрытым оверхедом Keras,
        # плюс Playwright рендерит HTML и снимает скриншоты). Время и шаг печатаем явно.
        epoch_start = time.time()
        n_files = len(gui_files)
        print("\n[TestingCallback] epoch {} — evaluating on {} files from {}".format(
            epoch + 1, n_files, input_path), flush=True)

        with open(self.logs_path, 'a') as file_to_write:
            for idx, file in enumerate(gui_files, start=1):
                file_start = time.time()
                gui_name = file.replace(".gui", "")
                print("[TestingCallback]   [{}/{}] {} — predicting…".format(
                    idx, n_files, gui_name), flush=True)

                # Та же предобработка, что и при обучении (BGR + /255), иначе train/test skew
                evaluation_img = Utils.get_preprocessed_img(
                    "{}/{}.png".format(input_path, gui_name), IMAGE_SIZE)

                t0 = time.time()
                result, _ = sampler.predict_greedy(self.model, np.array([evaluation_img]))
                predict_secs = time.time() - t0
                result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")
                print("[TestingCallback]   [{}/{}] predicted in {:.1f}s, len={}".format(
                    idx, n_files, predict_secs, len(result)), flush=True)

                if len(''.join(result.replace('\n', ''))) != 0:
                    resultBleu = BLEU.get_bleu_score(result, gui_name, input_path)
                    resultChrf = BLEU.get_chrf_score(result, gui_name, input_path)

                    print('{}'.format(file), flush=True)
                    print('BLEU score: {}'.format(resultBleu[0]), flush=True)
                    print('Individual 1-gram: %f' % resultBleu[1], flush=True)
                    print('Individual 2-gram: %f' % resultBleu[2], flush=True)
                    print('Individual 3-gram: %f' % resultBleu[3], flush=True)
                    print('Individual 4-gram: %f' % resultBleu[4], flush=True)
                    print('chrF score: %f' % resultChrf, flush=True)

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score: {}\n'.format(resultBleu[0]))
                    file_to_write.write('Individual 1-gram: %f\n' % resultBleu[1])
                    file_to_write.write('Individual 2-gram: %f\n' % resultBleu[2])
                    file_to_write.write('Individual 3-gram: %f\n' % resultBleu[3])
                    file_to_write.write('Individual 4-gram: %f\n' % resultBleu[4])
                    file_to_write.write('chrF score: %f\n' % resultChrf)
                    file_to_write.flush()

                    # На ранних эпохах предсказания обычно невалидны как DSL (несбалансированные
                    # теги и т.п.). Не даём одному плохому сэмплу уронить всё обучение —
                    # ловим исключение и записываем 100% diff как индикатор провала.
                    print("[TestingCallback]   [{}/{}] rendering+diff…".format(
                        idx, n_files), flush=True)
                    t1 = time.time()
                    try:
                        diff_master_and_prediction, diff_percentage = functional_test_instance.run_tests(result, gui_name)
                    except Exception as e:
                        print('Functional test failed on {}: {}: {}'.format(
                            gui_name, type(e).__name__, e), flush=True)
                        diff_percentage = 100
                    diff_secs = time.time() - t1

                    print('Image diff: {} (render+diff {:.1f}s)'.format(diff_percentage, diff_secs), flush=True)
                    file_to_write.write('Image diff: {}\n'.format(diff_percentage))
                    file_to_write.write('\n\n')
                    file_to_write.flush()

                else:
                    print('{}'.format(file), flush=True)
                    print('BLEU score -> {}'.format(0), flush=True)
                    print('chrf score -> {}'.format(0), flush=True)

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score: {}\n'.format(0))
                    file_to_write.write('chrf score: {}\n'.format(0))

                    print('Image diff -> {}'.format(100), flush=True)
                    file_to_write.write('Image diff: {}\n'.format(100))
                    file_to_write.write('\n\n')
                    file_to_write.flush()

                print("[TestingCallback]   [{}/{}] done in {:.1f}s".format(
                    idx, n_files, time.time() - file_start), flush=True)

        print("[TestingCallback] epoch {} eval done in {:.1f}s".format(
            epoch + 1, time.time() - epoch_start), flush=True)
