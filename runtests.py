
#!/usr/bin/env python3
"""Run the regression tests for imago."""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run imago regression tests")
    parser.add_argument(
        "--all",
        action="store_true",
        help="include directories named 'skip'",
    )
    parser.add_argument("paths", nargs="*", help="test files or directories to run")
    return parser.parse_args()


def find_imago_command(root: Path) -> Path:
    return root / "src" / "imago.py"


def collect_inputs(root_paths: list[Path], include_skip: bool) -> list[Path]:
    inputs: list[Path] = []

    def walk(current: Path) -> None:
        for child in sorted(current.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                if child.name == "skip" and not include_skip:
                    continue
                walk(child)
            elif child.is_file() and child.suffix.lower() in {".jpg", ".png"}:
                inputs.append(child)

    for path in root_paths:
        if path.is_file():
            if path.suffix.lower() in {".jpg", ".png"}:
                inputs.append(path)
        elif path.is_dir():
            walk(path)

    return sorted(inputs, key=lambda p: str(p))


def run_test(input_path: Path, imago_script: Path, root: Path) -> bool:
    print(f"running test: {input_path}")
    expected_path = input_path.with_suffix(".txt")
    if not expected_path.is_file():
        print("no expected-output file found")
        return True

    output_path = input_path.with_suffix(input_path.suffix + ".out")
    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        [sys.executable, str(imago_script), "--rng-seed", "runtests rng seed", str(input_path)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print("ERROR: imago returned failure")
        return False

    output_path.write_text(result.stdout, encoding="utf-8")
    if not expected_path.exists():
        print("no expected-output file found")
        return True

    expected_text = expected_path.read_text(encoding="utf-8")
    output_text = output_path.read_text(encoding="utf-8")
    if expected_text != output_text:
        print("ERROR: imago produced unexpected output")
        print("".join(difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            output_text.splitlines(keepends=True),
            fromfile=str(expected_path),
            tofile=str(output_path),
        )))
        return False

    output_path.unlink(missing_ok=True)
    return True


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    imago_script = find_imago_command(root)

    search_paths = [root / "tests"] if not args.paths else [Path(p).resolve() for p in args.paths]
    inputs = collect_inputs(search_paths, include_skip=args.all)

    failed = []
    for input_path in inputs:
        if not run_test(input_path, imago_script, root):
            failed.append(str(input_path))

    if failed:
        print("tests failed:")
        for item in failed:
            print(f"    {item}")
        return 1

    return 0


if __name__ == "__main__":
    main()