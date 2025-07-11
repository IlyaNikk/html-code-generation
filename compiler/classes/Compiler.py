import json
from .Node import *


class Compiler:
    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        self.root = Node("body", None, self.content_holder)

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        dsl_file = open(input_file_path)
        current_parent = self.root

        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)

    def compile_in_runtime(self, prediction, rendering_function=None):
        self.root = Node("body", None, self.content_holder)
        current_parent = self.root

        prediction_lines = prediction.splitlines()
        if prediction_lines[0].find("body") != -1:
            prediction_lines = prediction_lines[1:len(prediction_lines)-1]

        for token in prediction_lines:
            token = token.replace(" ", "").replace("\n", "").replace("\t", "")
            if len(token) == 0:
                continue

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                if token.find(",") != -1:
                    tokens = token.split(",")
                    for t in tokens[:-1]:
                        element = Node(t, current_parent, self.content_holder)
                        current_parent.add_child(element)

                    token = tokens[-1]
                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)

        return output_html
