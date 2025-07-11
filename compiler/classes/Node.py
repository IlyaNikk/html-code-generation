from __future__ import print_function


class Node:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        if self.key in mapping:
            value = mapping[self.key]
            if rendering_function is not None:
                value = rendering_function(self.key, value)

            if len(self.children) != 0:
                value = value.replace(self.content_holder, content)

            return value

        return ''
