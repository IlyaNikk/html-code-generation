import sys
import os
import re

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from classes.model.Main_Model import *

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

files = os.listdir(input_path)
gui_files = list(filter(lambda s: re.search(".*\\.gui$", s), files))
png_files = list(filter(lambda s: re.search(".*\\.png$", s), files))

# if len(gui_files) != len(png_files):
#     print("Error")
#     return ([])

print(gui_files)
for file in gui_files:
    resultBleu = BLEU.get_bleu_score(model, sampler, file, input_path)

    print(file)
    print('BLEU score -> {}'.format(resultBleu[0]))
    print('Individual 1-gram: %f' % resultBleu[1])
    print('Individual 2-gram: %f' % resultBleu[2])
    print('Individual 3-gram: %f' % resultBleu[3])
    print('Individual 4-gram: %f' % resultBleu[4])
