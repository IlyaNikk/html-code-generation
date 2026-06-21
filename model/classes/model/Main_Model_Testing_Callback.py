import os
import re
import sys
import keras.src.callbacks

sys.path.append('./')

from ..Sampler import *
from ..Utils import Utils
from ..dataset import profiles as dataset_profiles
from .Config import CONTEXT_LENGTH, IMAGE_SIZE
from compiler.classes.Compiler import *
from ..test_classes.Functional_Test import *
from ..test_classes.BLEU import *

# Веса (а значит и активный профиль датасета) живут здесь. DSL_PATH и input_path
# теперь резолвятся через profiles.load() — никаких хардкодов под конкретный сет.
trained_weights_path = "bin/web"
logs_path = "resources/logs.txt"


class TestingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
        input_shape = meta_dataset[0]
        output_size = meta_dataset[1]

        # Resolve dataset-specific paths from the sidecar written by train.py.
        profile = dataset_profiles.load(trained_weights_path)
        dsl_path = profile["dsl_mapping"]
        input_path = profile["eval_set"]

        sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)
        compiler = Compiler(dsl_path)

        functional_test_instance = FunctionalTest(self.model, sampler, compiler, input_path)

        # Берём первые 10 .gui-файлов из eval-сета активного профиля. Раньше тут
        # был хардкод 10 UUID из оригинального датасета, из-за чего после переключения
        # на синтетический сет callback падал с FileNotFoundError.
        all_gui = sorted(f for f in os.listdir(input_path) if re.search(r"\.gui$", f))
        gui_files = all_gui[:10]

        with open(logs_path, 'a') as file_to_write:
            for file in gui_files:
                gui_name = file.replace(".gui", "")
                # Та же предобработка, что и при обучении (BGR + /255), иначе train/test skew
                evaluation_img = Utils.get_preprocessed_img(
                    "{}/{}.png".format(input_path, gui_name), IMAGE_SIZE)

                result, _ = sampler.predict_greedy(self.model, np.array([evaluation_img]))
                result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")

                # print('test: {}, {}'.format(''.join(result.replace('\n', '')), len(''.join(result.replace('\n', '')))))
                if len(''.join(result.replace('\n', ''))) != 0:
                    resultBleu = BLEU.get_bleu_score(result, gui_name, input_path)
                    resultChrf = BLEU.get_chrf_score(result, gui_name, input_path)

                    print('{}'.format(file))
                    print('BLEU score: {}'.format(resultBleu[0]))
                    print('Individual 1-gram: %f' % resultBleu[1])
                    print('Individual 2-gram: %f' % resultBleu[2])
                    print('Individual 3-gram: %f' % resultBleu[3])
                    print('Individual 4-gram: %f' % resultBleu[4])
                    print('chrF score: %f' % resultChrf)

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score: {}\n'.format(resultBleu[0]))
                    file_to_write.write('Individual 1-gram: %f\n' % resultBleu[1])
                    file_to_write.write('Individual 2-gram: %f\n' % resultBleu[2])
                    file_to_write.write('Individual 3-gram: %f\n' % resultBleu[3])
                    file_to_write.write('Individual 4-gram: %f\n' % resultBleu[4])
                    file_to_write.write('chrF score: %f\n' % resultChrf)

                    # На ранних эпохах предсказания обычно невалидны как DSL (несбалансированные
                    # теги и т.п.). Не даём одному плохому сэмплу уронить всё обучение —
                    # ловим исключение и записываем 100% diff как индикатор провала.
                    try:
                        diff_master_and_prediction, diff_percentage = functional_test_instance.run_tests(result, gui_name)
                    except Exception as e:
                        print('Functional test failed on {}: {}: {}'.format(gui_name, type(e).__name__, e))
                        diff_percentage = 100

                    print('Image diff: {}'.format(diff_percentage))
                    file_to_write.write('Image diff: {}\n'.format(diff_percentage))
                    file_to_write.write('\n\n')

                else:
                    print('{}'.format(file))
                    print('BLEU score -> {}'.format(0))
                    print('chrf score -> {}'.format(0))

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score: {}\n'.format(0))
                    file_to_write.write('chrf score: {}\n'.format(0))

                    print('Image diff -> {}'.format(100))
                    file_to_write.write('Image diff: {}\n'.format(100))
                    file_to_write.write('\n\n')
