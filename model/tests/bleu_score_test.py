import sys
import os
import re

sys.path.append('./')

from model.classes.model.Main_Model import *

argv = sys.argv[1:]

if len(argv) < 3:
    print("Error: not enough argument supplied:")
    print("sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: "
          "greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
print("meta_dataset: ", meta_dataset)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = Main_Model(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

print(input_path)
files = os.listdir(input_path)
gui_files = list(filter(lambda s: re.search(".*\\.gui$", s), files))
png_files = list(filter(lambda s: re.search(".*\\.png$", s), files))

# if len(gui_files) != len(png_files):
#     print("Error")
#     return ([])

print(gui_files)
for file in gui_files:
    gui_name = file.replace(".gui", "")

    img = tf.keras.utils.load_img(
        "{}/{}.png".format(input_path, gui_name), target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    evaluation_img = tf.keras.utils.img_to_array(img)

    result, _ = sampler.predict_greedy(model, np.array([evaluation_img]), while_testing=True)
    print('result: {}'.format(result))

    result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")

    resultBleu = BLEU.get_bleu_score(result, gui_name, input_path)
    resultChrf = BLEU.get_chrf_score(result, gui_name, input_path)

    print(file)
    print('BLEU score -> {}'.format(resultBleu[0]))
    print('Individual 1-gram: %f' % resultBleu[1])
    print('Individual 2-gram: %f' % resultBleu[2])
    print('Individual 3-gram: %f' % resultBleu[3])
    print('Individual 4-gram: %f' % resultBleu[4])
    print('chrF score: %f' % resultBleu[4])
