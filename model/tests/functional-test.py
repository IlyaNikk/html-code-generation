import sys
import os
import re

sys.path.append('./')

from model.classes.model.Main_Model import *
from model.classes.test_classes.Functional_Test import *
from compiler.classes.Compiler import *

argv = sys.argv[1:]

dsl_path = "compiler/assets/web-dsl-mapping.json"

if len(argv) < 3:
    print("Error")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]
    save_images = False if len(argv) < 4 else True if int(argv[3]) == 1 else False

files = os.listdir(input_path)
gui_files = list(filter(lambda s: re.search(".*\\.gui$", s), files))
png_files = list(filter(lambda s: re.search(".*\\.png$", s), files))

if len(gui_files) != len(png_files):
    print("Error")
    exit(0)

gui_files = gui_files[:10]

# load model params
meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)

input_shape = meta_dataset[0]
output_size = meta_dataset[1]
model = Main_Model(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)
compiler = Compiler(dsl_path)

functional_test_instance = FunctionalTest(model, sampler, compiler, input_path)

for file in gui_files:
    gui_name = file.replace(".gui", "")

    img = tf.keras.utils.load_img(
        "{}/{}.png".format(input_path, gui_name), target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    evaluation_img = tf.keras.utils.img_to_array(img)

    result, _ = sampler.predict_greedy(model, np.array([evaluation_img]), while_testing=True)
    print(result)
    result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")
    diff_master_and_prediction, diff_percentage = functional_test_instance.run_tests(result, file)

    # save diff as image (optional)
    diff = Image.fromarray(diff_master_and_prediction, 'RGB')

    print('Image diff -> {}'.format(diff_percentage))

    if save_images:
        diff.save('diff_{}.png'.format(file.replace(".gui", "")))
