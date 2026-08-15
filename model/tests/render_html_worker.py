"""Render one already-compiled HTML document in an isolated WebKit process."""

import argparse
import sys

from playwright.sync_api import sync_playwright


DEFAULT_VIEWPORT = {"width": 1280, "height": 2860}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HTML file to render")
    parser.add_argument("--output", required=True, help="PNG destination")
    parser.add_argument("--page-timeout-seconds", type=int, default=90)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.page_timeout_seconds <= 0:
        raise SystemExit("--page-timeout-seconds must be positive")
    with open(args.input) as source:
        html = source.read()
    with sync_playwright() as playwright:
        browser = playwright.webkit.launch()
        try:
            # Reward scoring compares static HTML.  Bootstrap's carousel
            # script is irrelevant to that comparison and can keep expanding
            # state for pages with many carousel items (notably 90a52312…).
            # A JS-disabled context retains external CSS and the static DOM
            # while preventing that unbounded dynamic work.
            context = browser.new_context(viewport=DEFAULT_VIEWPORT, java_script_enabled=False)
            try:
                page = context.new_page()
                page.set_default_timeout(args.page_timeout_seconds * 1000)
                page.set_content(html, wait_until="domcontentloaded")
                page.screenshot(path=args.output)
            finally:
                context.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
