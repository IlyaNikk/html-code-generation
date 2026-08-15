import argparse
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
DEFAULT_OUTPUT_DIRECTORY = '{}/datasets/generated/web/article'.format(os.getcwd())
RENDER_TIMEOUT_MS = 60000
BLANK_IMAGE_MAX_CHANNEL_VALUE = 250

OUTPUT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY
FAILED_DIRECTORY = None
FAILED_LOG_PATH = None


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate synthetic screenshot/DSL pairs with an optional reproducible seed."
    )
    parser.add_argument("count", nargs="?", type=int, help="number of successful samples to generate")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerender-blank", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.retry_failed and args.rerender_blank:
        parser.error("--retry-failed and --rerender-blank are mutually exclusive")
    if not args.retry_failed and not args.rerender_blank:
        if args.count is None or args.count <= 0:
            parser.error("count must be a positive integer when generating samples")
    elif args.count is not None:
        parser.error("count cannot be used with --retry-failed or --rerender-blank")
    return args

def update_progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s status:%s\r' % (bar, percents, '%', '{}/{}'.format(count, total)))
    sys.stdout.flush()

class Generate_Dataset():
    def __init__(self, name=None):
        self.result = ''
        self.result_tree = []
        self.name = name or uuid.uuid4()
        self.compiler = Compiler(dsl_mapping_file_path)

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


def deterministic_name(seed, attempt):
    return uuid.uuid5(uuid.NAMESPACE_URL, "web-generated:{}:{}".format(seed, attempt))


def write_generation_manifest(args, successful_samples, attempts):
    manifest_path = os.path.join(OUTPUT_DIRECTORY, "generation_manifest.json")
    with open(manifest_path, "w") as destination:
        json.dump({
            "generator": "compiler/generate_dataset.py",
            "seed": args.seed,
            "requested_samples": args.count,
            "successful_samples": successful_samples,
            "attempts": attempts,
        }, destination, indent=2)
        destination.write("\n")
    print("[generate_dataset] manifest={}".format(manifest_path))


def main(argv):
    global OUTPUT_DIRECTORY, FAILED_DIRECTORY, FAILED_LOG_PATH
    args = parse_args(argv)
    OUTPUT_DIRECTORY = os.path.abspath(args.output_dir)
    FAILED_DIRECTORY = '{}/failed'.format(OUTPUT_DIRECTORY)
    FAILED_LOG_PATH = '{}/failed_samples.log'.format(OUTPUT_DIRECTORY)
    random.seed(args.seed)

    if args.retry_failed:
        retry_failed_samples()
        return
    if args.rerender_blank:
        rerender_blank_samples()
        return

    if os.path.isdir(OUTPUT_DIRECTORY) and os.listdir(OUTPUT_DIRECTORY):
        raise SystemExit(
            "refusing to mix a new dataset into non-empty output directory: {}".format(OUTPUT_DIRECTORY)
        )

    update_progress(0, args.count)
    successful_samples = []
    success_count = 0
    attempt_count = 0
    max_attempts = args.count * 3

    while success_count < args.count and attempt_count < max_attempts:
        attempt_count += 1
        name = deterministic_name(args.seed, attempt_count) if args.seed is not None else None
        generateModel = Generate_Dataset(name=name)
        generateModel.generate()
        generateModel.convert_to_string()

        if generateModel.generate_picture():
            generateModel.get_final_result()
            successful_samples.append(str(generateModel.name))
            success_count += 1
        else:
            print('[generate_dataset] skipped failed sample {}; saved to {}'.format(
                generateModel.name, FAILED_DIRECTORY), flush=True)

        update_progress(success_count, args.count)

    print('\n[generate_dataset] generated {} successful samples after {} attempts'.format(
        success_count, attempt_count))
    write_generation_manifest(args, successful_samples, attempt_count)

    if success_count < args.count:
        print('[generate_dataset] stopped before target after too many failed attempts; failed samples are in {}'.format(
            FAILED_DIRECTORY))
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
