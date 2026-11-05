"""Drive the running app (default http://127.0.0.1:7861) with the installed Chrome via Playwright and save screenshots:
  paper/figs/app_results.png    after clicking 'Run screening' on the built-in example
  paper/figs/app_benchmark.png  the live inference benchmark tab after a run
  paper/figs/app_invalid.png    an empty note -> input-problem message (recovery demo)
Start the app first:  PORT=7861 python app/app.py      then:  python src/make_screenshots.py"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861/"
OUT = Path(__file__).resolve().parents[1] / "paper" / "figs"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.get_by_role("button", name="Run screening").wait_for(timeout=60000)

    # 1. successful run
    page.get_by_role("button", name="Run screening").click()
    page.get_by_text("Overall preliminary outcome").wait_for(timeout=180000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "app_results.png"), full_page=True)
    print("saved app_results.png")

    # 2. live benchmark tab
    page.get_by_role("tab", name="Inference benchmark (live)").click()
    page.get_by_role("button", name="Run benchmark").click()
    page.get_by_text("Speculative (k=4)").first.wait_for(timeout=300000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "app_benchmark.png"), full_page=True)
    print("saved app_benchmark.png")

    # 3. invalid input and recovery
    page.get_by_role("tab", name="Screening").click()
    note = page.locator("textarea").first
    original = note.input_value()
    note.fill("")
    page.get_by_role("button", name="Run screening").click()
    page.get_by_text("Input problem").wait_for(timeout=60000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / "app_invalid.png"), full_page=True)
    print("saved app_invalid.png")
    note.fill(original)
    page.get_by_role("button", name="Run screening").click()
    page.get_by_text("Overall preliminary outcome").wait_for(timeout=180000)
    print("recovered after fixing the input: OK")
    browser.close()
