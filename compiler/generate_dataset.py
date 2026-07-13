import json
import random
import os
import uuid
import sys
import datetime
from PIL import Image, ImageStat
from playwright.sync_api import sync_playwright

sys.path.append('./')

from compiler.classes.Compiler import *
from compiler.classes.Utils import *

dsl_mapping_file_path = "compiler/assets/web-dsl-mapping-new.json"
dsl_mapping_rule_path = "compiler/assets/rules-for-generate.json"
OPEN_TAG_SLUG = "opening-tag"
CLOSE_TAG_SLUG = "closing-tag"
NEW_LINE = "\n"
ALL_CHILDREN_RESTRICTION = "allChildren"
ONLY_ONE_CHILD_RESTRICTION = "onlyOneChild"
SKIP_RESTRICTION = "skip"
OUTPUT_DIRECTORY = '{}/datasets/generated/web/article'.format(os.getcwd())
FAILED_DIRECTORY = '{}/failed'.format(OUTPUT_DIRECTORY)
FAILED_LOG_PATH = '{}/failed_samples.log'.format(OUTPUT_DIRECTORY)
RENDER_TIMEOUT_MS = 60000
BLANK_IMAGE_MAX_CHANNEL_VALUE = 250

argv = sys.argv[1:]

if len(argv) < 1:
    print("Error")
    exit(0)
elif argv[0] == "--retry-failed":
    count_iterations = 0
    retry_failed_mode = True
    rerender_blank_mode = False
elif argv[0] == "--rerender-blank":
    count_iterations = 0
    retry_failed_mode = False
    rerender_blank_mode = True
else:
    count_iterations = int(argv[0])
    retry_failed_mode = len(argv) > 1 and argv[1] == "--retry-failed"
    rerender_blank_mode = len(argv) > 1 and argv[1] == "--rerender-blank"

def update_progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s status:%s\r' % (bar, percents, '%', '{}/{}'.format(count, total)))
    sys.stdout.flush()

class Generate_Dataset():
    def __init__(self):
        self.result = ''
        self.result_tree = []
        self.name = uuid.uuid4()
        self.compiler = Compiler(dsl_mapping_file_path)
        random.seed()

        with open(dsl_mapping_file_path) as data_file:
            self.dsl_elements = json.load(data_file)

        with open(dsl_mapping_rule_path) as data_file:
            dsl_all_rules = json.load(data_file)

        self.dsl_rules = dsl_all_rules["relations"]
        self.max_length = dsl_all_rules["length"]
        self.top_element = dsl_all_rules["top"]
        self.structure = dsl_all_rules["structure"]
        self.restrictions = dsl_all_rules["restrictions"]

        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

        except_child_array = [OPEN_TAG_SLUG, CLOSE_TAG_SLUG, self.top_element] + self.structure

        for elem in self.dsl_elements:
            if elem in self.restrictions and SKIP_RESTRICTION in self.restrictions[elem]:
                except_child_array.append(elem)
                continue

        for elem in self.dsl_elements:
            if elem not in self.dsl_rules:
                if elem in except_child_array:
                    continue

                self.dsl_rules[elem] = []
                for element_from_all_list in self.dsl_elements:
                    if element_from_all_list in except_child_array or element_from_all_list == elem:
                        continue
                    self.dsl_rules[elem].append(element_from_all_list)

    def convert_to_string(self):
        tab_count = 0

        for element in self.result_tree:
            if element == self.dsl_elements[OPEN_TAG_SLUG]:
                self.result = self.result + " " + element + NEW_LINE
                tab_count = tab_count + 1

                for i in range(tab_count):
                    self.result = self.result + "\t"
            elif element == self.dsl_elements[CLOSE_TAG_SLUG]:
                self.result = self.result + NEW_LINE
                tab_count = tab_count - 1
                for i in range(tab_count):
                    self.result = self.result + "\t"

                self.result = self.result + element + NEW_LINE

                for i in range(tab_count):
                    self.result = self.result + "\t"

            else:
                self.result = self.result + " " + element

    def generate_child(self, parent, count=0):
        children = []

        if len(self.dsl_rules[parent]) != 0:
            if parent in self.restrictions and ALL_CHILDREN_RESTRICTION in self.restrictions[parent]:
                child_count = random.randint(1, self.max_length / 10)
                for child in self.dsl_rules[parent]:
                    children.append(child)

                    children.append(self.dsl_elements[OPEN_TAG_SLUG])
                    children = children + self.generate_child(child, child_count)
                    children.append(self.dsl_elements[CLOSE_TAG_SLUG])

                return children

            count = count or random.randint(1, self.max_length / 10)
            for i in range(count):
                child = random.choice(self.dsl_rules[parent])

                children.append(child)

                if len(self.dsl_rules[child]) != 0:
                    children.append(self.dsl_elements[OPEN_TAG_SLUG])
                    children = children + self.generate_child(child)
                    children.append(self.dsl_elements[CLOSE_TAG_SLUG])
                else:
                    if i != count - 1:
                        children.append(",")

        return children

    def generate(self):
        self.result_tree.append(self.top_element)
        self.result_tree.append(self.dsl_elements[OPEN_TAG_SLUG])

        for elem in self.structure:
            self.result_tree.append(elem)
            self.result_tree.append(self.dsl_elements[OPEN_TAG_SLUG])
            self.result_tree = self.result_tree + self.generate_child(elem)
            self.result_tree.append(self.dsl_elements[CLOSE_TAG_SLUG])

        self.result_tree.append(self.dsl_elements[CLOSE_TAG_SLUG])

    def get_final_result(self):
        if self.result == '':
            self.convert_to_string()

        with open('{}/{}.gui'.format(OUTPUT_DIRECTORY, self.name), 'a') as file_to_write:
        # with open('{}/file_name.gui'.format(OUTPUT_DIRECTORY), 'a') as file_to_write:
            file_to_write.write(self.result)

        return self.result

    def save_failed_result(self, error):
        if not os.path.exists(FAILED_DIRECTORY):
            os.makedirs(FAILED_DIRECTORY)

        with open('{}/{}.gui'.format(FAILED_DIRECTORY, self.name), 'w') as file_to_write:
            file_to_write.write(self.result)

        with open(FAILED_LOG_PATH, 'a') as failed_log:
            failed_log.write('{}\t{}\t{}\n'.format(
                datetime.datetime.now().isoformat(),
                self.name,
                error
            ))

    @staticmethod
    def is_blank_screenshot(path):
        with Image.open(path) as screenshot:
            screenshot = screenshot.convert('RGB')
            extrema = ImageStat.Stat(screenshot).extrema
            return all(channel_min >= BLANK_IMAGE_MAX_CHANNEL_VALUE for channel_min, _ in extrema)

    def generate_picture(self):
        master_html = self.compiler.compile_in_runtime(self.result, rendering_function=Utils.render_content_with_random_text)
        screenshot_path = '{}/{}.png'.format(OUTPUT_DIRECTORY, self.name)

        # with open('{}/{}.html'.format(OUTPUT_DIRECTORY, self.name), 'a') as file_to_write:
        #     file_to_write.write(master_html)

        try:
            with sync_playwright() as p:
                browser = p.webkit.launch()
                page = browser.new_page()
                #
                page.set_viewport_size({"width": 1280, "height": 2860})
                page.set_content(master_html, wait_until="load")
                page.screenshot(path=screenshot_path)

                if self.is_blank_screenshot(screenshot_path):
                    os.remove(screenshot_path)
                    raise RuntimeError("blank screenshot")

                browser.close()
                return True
        except Exception as e:
            last_error = '{}: {}'.format(type(e).__name__, e)
            if os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                except OSError:
                    pass
            print('\n[generate_dataset] screenshot failed for {}; skipping GUI and generating a new one: {}'.format(
                self.name, last_error), flush=True)
            self.save_failed_result(last_error)
            return False


def retry_failed_samples():
    if not os.path.isdir(FAILED_DIRECTORY):
        print('[generate_dataset] no failed samples directory: {}'.format(FAILED_DIRECTORY))
        return

    failed_files = [
        f for f in os.listdir(FAILED_DIRECTORY)
        if f.endswith('.gui')
    ]

    if len(failed_files) == 0:
        print('[generate_dataset] no failed .gui files to retry in {}'.format(FAILED_DIRECTORY))
        return

    success_count = 0
    for failed_file in failed_files:
        generateModel = Generate_Dataset()
        generateModel.name = failed_file.replace('.gui', '')

        failed_path = '{}/{}'.format(FAILED_DIRECTORY, failed_file)
        with open(failed_path) as source:
            generateModel.result = source.read()

        if generateModel.generate_picture():
            with open('{}/{}.gui'.format(OUTPUT_DIRECTORY, generateModel.name), 'w') as target:
                target.write(generateModel.result)
            os.rename(failed_path, '{}.done'.format(failed_path))
            success_count += 1
            print('[generate_dataset] retried successfully: {}'.format(generateModel.name), flush=True)
        else:
            print('[generate_dataset] retry still failed: {}'.format(generateModel.name), flush=True)

    print('[generate_dataset] retried {} of {} failed samples'.format(success_count, len(failed_files)))


def rerender_blank_samples():
    gui_files = [
        f for f in os.listdir(OUTPUT_DIRECTORY)
        if f.endswith('.gui') and os.path.isfile('{}/{}'.format(OUTPUT_DIRECTORY, f))
    ]

    if len(gui_files) == 0:
        print('[generate_dataset] no .gui files to rerender in {}'.format(OUTPUT_DIRECTORY))
        return

    rerendered_count = 0
    failed_count = 0
    for gui_file in gui_files:
        sample_name = gui_file.replace('.gui', '')
        screenshot_path = '{}/{}.png'.format(OUTPUT_DIRECTORY, sample_name)

        needs_rerender = not os.path.isfile(screenshot_path)
        if not needs_rerender:
            try:
                needs_rerender = Generate_Dataset.is_blank_screenshot(screenshot_path)
            except Exception:
                needs_rerender = True

        if not needs_rerender:
            continue

        generateModel = Generate_Dataset()
        generateModel.name = sample_name
        with open('{}/{}'.format(OUTPUT_DIRECTORY, gui_file)) as source:
            generateModel.result = source.read()

        if generateModel.generate_picture():
            rerendered_count += 1
            print('[generate_dataset] rerendered: {}'.format(sample_name), flush=True)
        else:
            failed_count += 1
            print('[generate_dataset] rerender failed: {}'.format(sample_name), flush=True)

    print('[generate_dataset] rerendered {} samples; {} failed'.format(
        rerendered_count, failed_count))


if retry_failed_mode:
    retry_failed_samples()
    exit(0)

if rerender_blank_mode:
    rerender_blank_samples()
    exit(0)

update_progress(0, count_iterations)

success_count = 0
attempt_count = 0
max_attempts = count_iterations * 3

while success_count < count_iterations and attempt_count < max_attempts:
    attempt_count += 1
    generateModel = Generate_Dataset()
    generateModel.generate()
    generateModel.convert_to_string()

    if generateModel.generate_picture():
        generateModel.get_final_result()
        success_count += 1
    else:
        print('[generate_dataset] skipped failed sample {}; saved to {}'.format(
            generateModel.name, FAILED_DIRECTORY), flush=True)

    update_progress(success_count, count_iterations)

print('\n[generate_dataset] generated {} successful samples after {} attempts'.format(
    success_count, attempt_count))

if success_count < count_iterations:
    print('[generate_dataset] stopped before target after too many failed attempts; failed samples are in {}'.format(
        FAILED_DIRECTORY))
