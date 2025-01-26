import sys
import os
import re
import struct

import tensorflow as tf
import numpy as np

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import sync_playwright

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from model.classes.Sampler import *
from model.classes.model.Main_Model import *
from compiler.classes.Compiler import *
from compiler.classes.Utils import *

TEXT_PLACE_HOLDER = "[]"


def render_content_with_text(key, value):
    if key.find("btn") != -1:
        return value.replace(TEXT_PLACE_HOLDER, "What is Lorem?")
    elif key.find("title") != -1:
        return value.replace(TEXT_PLACE_HOLDER, "Lorem Ipsum")
    elif key.find("text") != -1:
        return value.replace(TEXT_PLACE_HOLDER, "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                                                " Lorem Ipsum has been the industry's standard dummy text ever since the 1500s")
    return value


argv = sys.argv[1:]

if len(argv) < 3:
    print("Error")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]

files = os.listdir(input_path)
# html_files = filter(lambda s: re.search(".*\\.html$", s), files)
# png_files = filter(lambda s: re.search(".*\\.png$", s), files)
#
# if len(html_files) != len(png_files):
#     print("Error")
#     exit(0)

html_files = ['0CC0512B-11C5-481C-BC81-534F1FC9EC0A.gui']

# load model params
meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
print("meta_dataset: ", meta_dataset)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]
model = Main_Model(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

dsl_path = "../compiler/assets/web-dsl-mapping.json"
compiler = Compiler(dsl_path)

for file in html_files:
    file_name = file.replace(".gui", "")
    master_gui = open(file, 'r').read()
    master_gui = master_gui.replace(START_TOKEN, "").replace(END_TOKEN, "")

# get diff between images
    # diff_array = img1_array - img2_array

    # print(np.sum(np.abs(diff_array)))
    # print(np.sum(np.abs(img1_array)))
    # print('diff {}'.format(np.sum(np.abs(diff_array)) * 100 / np.sum(np.abs(img1_array))))

    img = tf.keras.utils.load_img(
        "{}/{}.png".format(input_path, file_name), target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    evaluation_img = tf.keras.utils.img_to_array(img)

    result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
    result = result.replace(START_TOKEN, "").replace(END_TOKEN, "")

    master_html = compiler.compile_in_runtime(master_gui, rendering_function=render_content_with_text)
    compiled_after_prediction_html = compiler.compile_in_runtime(result, rendering_function=render_content_with_text)

    with sync_playwright() as p:
        browser = p.webkit.launch()
        page = browser.new_page()
        # брать html и брать новый html
        page.set_viewport_size({"width": 1280, "height": 986})
        page.set_content(master_html, wait_until="load")
        page.screenshot(path='master_screenshot.png')

        # page.set_viewport_size({"width": 1280, "height": 986})
        page.set_content(compiled_after_prediction_html, wait_until="load")
        page.screenshot(path='prediction_screenshot.png')
        browser.close()

    with Image.open('master_screenshot.png') as html_master_screenshot:
        html_master_screenshot.load()
        html_master_screenshot = html_master_screenshot.convert('RGB')

    with Image.open('prediction_screenshot.png') as html_prediction_screenshot:
        html_prediction_screenshot.load()
        html_prediction_screenshot = html_prediction_screenshot.convert('RGB')

    # convert images into array
    html_master_screenshot_array = np.float32(html_master_screenshot)
    html_master_screenshot_array_color = np.uint8(html_master_screenshot_array)

    html_prediction_screenshot_array = np.float32(html_prediction_screenshot)
    html_prediction_screenshot_array_color = np.uint8(html_prediction_screenshot_array)

    diff_master_and_prediction = np.subtract(html_master_screenshot_array_color ,html_prediction_screenshot_array_color)

    print('diff {}'.format(np.sum(np.abs(diff_master_and_prediction)) * 100 / np.sum(np.abs(html_prediction_screenshot_array_color))))

    # save diff as image (optional)
    diff = Image.fromarray(diff_master_and_prediction, 'RGB')
    diff.save('diff.png')
    diff.show()
