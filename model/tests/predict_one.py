import sys
import os

sys.path.append('./')

import numpy as np

from model.classes.model.Main_Model import *
from model.classes.Utils import Utils
from model.classes.dataset import profiles as dataset_profiles
from compiler.classes.Compiler import Compiler

# DSL_PATH теперь определяется по training_profile.txt в каталоге весов
# (см. classes/dataset/profiles.py). Раньше был хардкод под оригинальный
# датасет — после переобучения на синтетике он указывал не туда.
TEXT_PLACE_HOLDER = "[]"


def render_content_with_text(key, value):
    if key.find("btn") != -1:
        return value.replace(TEXT_PLACE_HOLDER, "What is Lorem?")
    if key.find("title") != -1:
        return value.replace(TEXT_PLACE_HOLDER, "Lorem Ipsum")
    if key.find("text") != -1:
        return value.replace(
            TEXT_PLACE_HOLDER,
            "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
            " Lorem Ipsum has been the industry's standard dummy text ever since the 1500s",
        )
    return value


def main():
    argv = sys.argv[1:]
    if len(argv) < 3:
        print("Usage: predict_one.py <weights_path> <model_name> <image_path> [output_dir]")
        sys.exit(1)

    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    image_path = argv[2]
    output_dir = argv[3] if len(argv) > 3 else "generated-output"

    if not os.path.isfile(image_path):
        print("Image not found: {}".format(image_path))
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
    input_shape = meta_dataset[0]
    output_size = meta_dataset[1]

    model = Main_Model(input_shape, output_size, trained_weights_path)
    model.load(trained_model_name)

    sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

    # Predict.
    # while_testing=True важно: Sampler.predict_greedy в этой ветке вызывает
    # Main_Model.predict(image, partial_caption) — обёртку с двумя позиционными
    # аргументами. Без этого флага он попытается вызвать его как сырой
    # Keras-Model.predict([...], batch_size=1, ...), и Main_Model.predict
    # упадёт с TypeError: unexpected keyword argument 'batch_size'.
    evaluation_img = Utils.get_preprocessed_img(image_path, IMAGE_SIZE)
    result, _ = sampler.predict_greedy(model, np.array([evaluation_img]), while_testing=True)
    gui_text = result.replace(START_TOKEN, "").replace(END_TOKEN, "")

    # Display + save DSL
    sep = "=" * 60
    print("\n" + sep)
    print("Predicted DSL for {}".format(image_path))
    print(sep)
    print(gui_text if gui_text.strip() else "<empty prediction>")
    print(sep)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    gui_out = os.path.join(output_dir, "{}.predicted.gui".format(stem))
    with open(gui_out, "w") as f:
        f.write(gui_text)
    print("  DSL:        {}".format(gui_out))

    # Compile to HTML using the DSL mapping recorded at training time.
    dsl_path = dataset_profiles.load(trained_weights_path)["dsl_mapping"]
    compiler = Compiler(dsl_path)
    html_out = os.path.join(output_dir, "{}.predicted.html".format(stem))
    html = None
    try:
        html = compiler.compile_in_runtime(gui_text, rendering_function=render_content_with_text)
        with open(html_out, "w") as f:
            f.write(html)
        print("  HTML:       {}".format(html_out))
    except Exception as e:
        print("  HTML:       FAILED ({}: {})".format(type(e).__name__, e))

    # Render screenshot
    if html:
        screenshot_out = os.path.join(output_dir, "{}.predicted.png".format(stem))
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.webkit.launch()
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 986})
                page.set_content(html, wait_until="load")
                page.screenshot(path=screenshot_out)
                browser.close()
            print("  Screenshot: {}".format(screenshot_out))
        except ImportError:
            print("  Screenshot: SKIPPED — playwright not installed "
                  "(pip install playwright && playwright install)")
        except Exception as e:
            print("  Screenshot: FAILED ({}: {})".format(type(e).__name__, e))


if __name__ == "__main__":
    main()
