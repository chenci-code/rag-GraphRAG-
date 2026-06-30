from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)

    if len(sys.argv) < 2:
        print("Usage: python scripts/deepseek_ocr.py <pdf_path>", file=sys.stderr)
        return 2

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    model = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    api_key = os.getenv("DEEPSEEK_OCR_API_KEY", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("DEEPSEEK_OCR_API_KEY or DEEPSEEK_API_KEY is missing in .env.", file=sys.stderr)
        return 2

    print(
        "DeepSeek-OCR adapter is configured but not implemented yet.\n"
        f"Model: {model}\n"
        f"Input: {pdf_path}\n\n"
        "请在 scripts/deepseek_ocr.py 中接入你实际安装的 deepseek-ai/DeepSeek-OCR "
        "SDK 或命令行，并把 OCR 结果打印到 stdout。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
