"""Browser end-to-end check of the web UI against a RUNNING server (python run.py), using the locally
installed Microsoft Edge through Playwright (dev-only: pip install playwright; no browser download).

    .venv\\Scripts\\python.exe scripts\\ui_e2e.py [--prompt "..."] [--shots DIR]
"""

import argparse
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

URL = "http://127.0.0.1:7860/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a small red sailboat on a calm turquoise lagoon, aerial photograph, midday sun")
    ap.add_argument("--shots", default="outputs/ui_e2e")
    a = ap.parse_args()
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL)
        expect(page.locator("#dims")).to_have_text("1024 × 1024")  # default 1:1, 1.0 MP, x16
        expect(page.locator("#gpu-name")).to_contain_text("NVIDIA", timeout=10000)

        # resolution controls
        page.select_option("#aspect", "16:9")
        expect(page.locator("#dims")).to_have_text("1360 × 768")
        page.select_option("#multiple", "64")
        expect(page.locator("#dims")).to_have_text("1344 × 768")
        page.select_option("#multiple", "16")

        # prompt + encode
        page.fill("#prompt", a.prompt)
        expect(page.locator("#encode-state")).to_have_text("Not Encoded")
        expect(page.locator("#generate-btn")).to_be_disabled()
        page.click("#encode-btn")
        expect(page.locator("#encode-state")).to_have_text("Encoding", timeout=5000)
        page.screenshot(path=str(shots / "1_encoding.png"))
        expect(page.locator("#encode-state")).to_have_text("Encoded", timeout=300000)
        expect(page.locator("#generate-btn")).to_be_enabled()

        # editing the prompt invalidates; restoring it re-validates
        page.fill("#prompt", a.prompt + " at sunset")
        expect(page.locator("#encode-state")).to_have_text("Not Encoded")
        expect(page.locator("#generate-btn")).to_be_disabled()
        page.fill("#prompt", a.prompt)
        expect(page.locator("#encode-state")).to_have_text("Encoded")

        # generate
        t0 = time.time()
        page.click("#generate-btn")
        expect(page.locator("#encode-btn")).to_be_disabled(timeout=5000)  # no overlapping GPU jobs
        expect(page.locator("#gen-stage")).to_have_text("Sampling", timeout=300000)
        page.screenshot(path=str(shots / "2_sampling.png"))
        expect(page.locator("#gen-status")).to_have_text("Complete", timeout=600000)
        img = page.locator("#image")
        expect(img).to_be_visible()
        page.wait_for_function("document.getElementById('image').naturalWidth > 0")
        nat = page.evaluate("[document.getElementById('image').naturalWidth, document.getElementById('image').naturalHeight]")
        box = img.bounding_box()
        print(f"image natural {nat}, displayed {box['width']:.0f}x{box['height']:.0f} (aspect {nat[0] / nat[1]:.4f} vs {box['width'] / box['height']:.4f}), {time.time() - t0:.0f}s")
        assert nat == [1360, 768] and abs(nat[0] / nat[1] - box["width"] / box["height"]) < 0.01
        print("info:", page.inner_text("#image-info"), "| seed shown:", page.inner_text("#gen-seed"))
        expect(page.locator("#download")).to_be_visible()
        expect(page.locator("#encode-state")).to_have_text("Encoded")  # conditioning preserved after generation
        page.screenshot(path=str(shots / "3_complete.png"))

        # settings page
        page.click("#open-settings")
        expect(page.locator("#info-transformer_path")).to_contain_text("fp8_scaled", timeout=10000)
        page.screenshot(path=str(shots / "4_settings.png"), full_page=True)
        page.click("#close-settings")
        print("page errors:", errors or "none")
        browser.close()


if __name__ == "__main__":
    main()
