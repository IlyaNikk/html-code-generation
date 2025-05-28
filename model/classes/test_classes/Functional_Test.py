import sys
import numpy as np
import tensorflow as tf

sys.path.append('./')

from PIL import Image
from playwright.sync_api import sync_playwright

from ..Vocabulary import START_TOKEN, END_TOKEN
from ..model.Config import IMAGE_SIZE
from compiler.classes.Utils import *

TEXT_PLACE_HOLDER = "[]"


class FunctionalTest:
    def __init__(self, model, sampler, compiler, input_path):
        self.model = model
        self.sampler = sampler
        self.compiler = compiler
        self.input_path = input_path

    def run_tests(self, predict_result, file_name):
        gui_name = file_name.replace(".gui", "")
        master_gui = open("{}/{}.gui".format(self.input_path, gui_name), 'r').read()
        master_gui = master_gui.replace(START_TOKEN, "").replace(END_TOKEN, "")

        result = predict_result.replace(START_TOKEN, "").replace(END_TOKEN, "")

        # print("Functional result: {}".format(result))

        master_html = self.compiler.compile_in_runtime(master_gui, rendering_function=Utils.render_content_with_text)
        compiled_after_prediction_html = self.compiler.compile_in_runtime(result, rendering_function=Utils.render_content_with_text)

        with sync_playwright() as p:
            browser = p.webkit.launch()
            page = browser.new_page()
            #
            page.set_viewport_size({"width": 1280, "height": 2860})
            page.set_content(master_html, wait_until="load")
            page.screenshot(path='master_screenshot.png')

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

        diff_master_and_prediction = np.subtract(html_master_screenshot_array_color, html_prediction_screenshot_array_color)

        diff_percentage = (np.sum(np.abs(diff_master_and_prediction)) * 100
                           / np.sum(np.abs(html_prediction_screenshot_array_color)))
        print('diff for {}: {}'.format(gui_name, diff_percentage))

        return diff_master_and_prediction, diff_percentage
