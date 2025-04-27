import json
import random
import os
import uuid
import sys
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
OUTPUT_DIRECTORY = '{}/datasets/generated'.format(os.getcwd())

argv = sys.argv[1:]

if len(argv) < 1:
    print("Error")
    exit(0)
else:
    count_iterations = int(argv[0])

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
        self.convert_to_string()

        with open('{}/{}.gui'.format(OUTPUT_DIRECTORY, self.name), 'a') as file_to_write:
        # with open('{}/file_name.gui'.format(OUTPUT_DIRECTORY), 'a') as file_to_write:
            file_to_write.write(self.result)

        return self.result

    def generate_picture(self):
        master_html = self.compiler.compile_in_runtime(self.result, rendering_function=Utils.render_content_with_random_text)

        # with open('{}/{}.html'.format(OUTPUT_DIRECTORY, self.name), 'a') as file_to_write:
        #     file_to_write.write(master_html)

        with sync_playwright() as p:
            browser = p.webkit.launch()
            page = browser.new_page()
            #
            page.set_viewport_size({"width": 1280, "height": 2860})
            page.set_content(master_html, wait_until="load")
            page.screenshot(path='{}/{}.png'.format(OUTPUT_DIRECTORY, self.name))

            browser.close()


update_progress(0, count_iterations)

for i in range(count_iterations):
    generateModel = Generate_Dataset()
    generateModel.generate()
    generateModel.get_final_result()
    generateModel.generate_picture()
    update_progress(i, count_iterations)
