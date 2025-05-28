from __future__ import print_function

__author__ = 'Taneem Jan, taneemishere.github.io'

import sys

from os.path import basename
from classes.Utils import *
from classes.Compiler import *

if __name__ == "__main__":
    argv = sys.argv[1:]
    length = len(argv)
    if length != 0:
        input_file = argv[0]
    else:
        print("Error: not enough argument supplied:")
        print("web-compiler.py <path> <file name>")
        exit(0)

FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

# dsl_path = "compiler/assets/web-dsl-mapping.json"
dsl_path = "compiler/assets/web-dsl-mapping-new.json"
compiler = Compiler(dsl_path)

file_uid = basename(input_file)[:basename(input_file).find(".")]
path = input_file[:input_file.find(file_uid)]

input_file_path = "{}{}.gui".format(path, file_uid)
output_file_path = "{}{}.html".format(path, file_uid)

# compiler.compile(input_file_path, output_file_path, rendering_function=Utils.render_content_with_text)
gui = open(input_file_path).read()
result = compiler.compile_in_runtime(gui, rendering_function=Utils.render_content_with_text)
print(output_file_path)

with open(output_file_path, 'w') as output_file:
    output_file.write(result)
