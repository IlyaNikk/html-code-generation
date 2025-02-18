import os
import sys
import re
import keras.src.callbacks
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("/Users/ivnikitin/Desktop/аспирантура/Научная работа/html-code-generation-from-images-with-deep-neural-networks")
print('/Users/ivnikitin/Desktop/аспирантура/Научная работа/html-code-generation-from-images-with-deep-neural-networks')

from ..Sampler import *
from .Config import CONTEXT_LENGTH
from compiler.classes.Compiler import *
from tests.test_classes.Functional_Test import *
from tests.test_classes.BLEU import *
from classes.model.Config import IMAGE_SIZE

DSL_PATH = "../compiler/assets/web-dsl-mapping.json"
trained_weights_path = "../bin/web"
input_path = "../datasets/web/eval_set"

class TestingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
        input_shape = meta_dataset[0]
        output_size = meta_dataset[1]
        print(self.params)

        sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)
        compiler = Compiler(DSL_PATH)

        functional_test_instance = FunctionalTest(self.model, sampler, compiler, input_path)

        # files = os.listdir(input_path)
        # gui_files = list(filter(lambda s: re.search(".*\\.gui$", s), files))
        # gui_files = gui_files[:10]

        gui_files = [
            "BD715F1F-4494-4A0C-8875-4334052D699B.gui",
            "3E106EF6-1841-4149-BC43-0D77B00A7241.gui",
            "1B50F242-FB10-495E-91CE-D2C2D0BB46C5.gui",
            "967A76E0-0B09-48D6-AD4A-5E40CE51FEFD.gui",
            "0529E00F-15B6-43BD-86FB-F15245861B07.gui",
            "CBE1F184-0AEC-4E32-BC38-00D7B7D5B284.gui",
            "0DF671EC-7BD9-4142-8A9D-5B438DD37323.gui",
            "B9261BA5-39AE-4261-9F43-29F25695C821.gui",
            "529820DE-98F5-4342-A20C-6B65580BECA7.gui",
            "2F2F9495-4422-4B87-9F30-260974449686.gui"
        ]

        with open('../resources/logs.txt', 'a') as file_to_write:
            for file in gui_files:
                gui_name = file.replace(".gui", "")
                img = tf.keras.utils.load_img(
                    "{}/{}.png".format(input_path, gui_name), target_size=(IMAGE_SIZE, IMAGE_SIZE)
                )
                evaluation_img = tf.keras.utils.img_to_array(img)

                result, _ = sampler.predict_greedy(self.model, np.array([evaluation_img]))
                result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")

                # print('test: {}, {}'.format(''.join(result.replace('\n', '')), len(''.join(result.replace('\n', '')))))
                if len(''.join(result.replace('\n', ''))) != 0:
                    resultBleu = BLEU.get_bleu_score(result, gui_name, input_path)

                    print('{}'.format(file))
                    print('BLEU score -> {}'.format(resultBleu[0]))
                    print('Individual 1-gram: %f' % resultBleu[1])
                    print('Individual 2-gram: %f' % resultBleu[2])
                    print('Individual 3-gram: %f' % resultBleu[3])
                    print('Individual 4-gram: %f' % resultBleu[4])

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score -> {}\n'.format(resultBleu[0]))
                    file_to_write.write('Individual 1-gram: %f\n' % resultBleu[1])
                    file_to_write.write('Individual 2-gram: %f\n' % resultBleu[2])
                    file_to_write.write('Individual 3-gram: %f\n' % resultBleu[3])
                    file_to_write.write('Individual 4-gram: %f\n' % resultBleu[4])

                    diff_master_and_prediction, diff_percentage = functional_test_instance.run_tests(result, gui_name)

                    print('Image diff -> {}'.format(diff_percentage))
                    file_to_write.write('Image diff -> {}\n'.format(diff_percentage))
                    file_to_write.write('\n\n')

                else:
                    print('{}'.format(file))
                    print('BLEU score -> {}'.format(0))

                    file_to_write.write('{} \n'.format(file))
                    file_to_write.write('BLEU score -> {}\n'.format(0))

                    print('Image diff -> {}'.format(100))
                    file_to_write.write('Image diff -> {}\n'.format(100))
                    file_to_write.write('\n\n')
