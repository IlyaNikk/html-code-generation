from __future__ import print_function
from __future__ import absolute_import
from difflib import SequenceMatcher
import glob

# Modified version
__author__ = 'Taneem Jan, taneemishere.github.io'


import sys

from os.path import basename
from classes.Sampler import *
from classes.model.Main_Model import *
import tensorflow as tf

argv = sys.argv[1:]

if len(argv) < 4:
    print("Error: not enough argument supplied:")
    print("sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: "
          "greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]
    output_path = argv[3]
    search_method = "greedy" if len(argv) < 5 else argv[4]

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
print("meta_dataset: ", meta_dataset)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = Main_Model(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

all_image_files = glob.glob(input_path + '/*.png')
all_gui_files = glob.glob(input_path + '/*.gui')
sum_sequence_ratio = [];

for file in all_image_files:
    file_name_img = basename(file)[:basename(file).find(".")]
    img = tf.keras.utils.load_img(
        file, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    evaluation_img = tf.keras.utils.img_to_array(img)

    if search_method == "greedy":
        result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
        # print("Result greedy: {}".format(result))

    print(file_name_img)
    correct_result = open(input_path + '/' + file_name_img +'.gui', 'r').read()
    current_seq = SequenceMatcher(None, result, correct_result)
    sum_sequence_ratio.append(current_seq.ratio())
    # print(sum_sequence_ratio)

# with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
#     out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
average = sum(sum_sequence_ratio) / len(sum_sequence_ratio)
print('Error result: ' + str(average))
