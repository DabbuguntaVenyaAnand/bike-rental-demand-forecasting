"""Render project markdown documents to PDF using headless Chrome.

Usage:
    .venv/Scripts/python scripts/md2pdf.py docs/person_a_writeup.md docs/team_writeup.md

Output: same path with .pdf extension.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #2c3e50; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #2c3e50; margin-top: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 11.5pt; color: #34495e; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 6pt 0; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
th { background: #eef2f5; }
code, pre { font-family: Consolas, monospace; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #ddd; padding: 6pt; overflow-wrap: anywhere; }
blockquote { border-left: 3px solid #2c3e50; margin-left: 0; padding-left: 8pt; color: #333; }
img { max-width: 100%; }
"""


def find_chrome() -> str:
    for p in CHROME_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("Chrome not found; install Google Chrome or edit CHROME_CANDIDATES.")


def convert(md_path: Path) -> Path:
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    out_pdf = md_path.with_suffix(".pdf").resolve()
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as fh:
        fh.write(html)
        html_path = fh.name
    result = subprocess.run(
        [
            find_chrome(),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            f"file:///{html_path.replace(chr(92), '/')}",
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or not out_pdf.exists():
        raise RuntimeError(
            f"Chrome failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )
    print(f"OK {md_path.name} -> {out_pdf.name} ({out_pdf.stat().st_size:,} bytes)")
    return out_pdf


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        convert(Path(arg))


if __name__ == "__main__":
    main()
