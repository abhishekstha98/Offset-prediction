import re
from pathlib import Path

RE_UNSEEN = re.compile(r"UNSEEN\s+mean MAE Tmax:\s*([\d.]+)\s*±\s*([\d.]+)")

with open("outputs/withholding_test.log") as f:
    for line in f:
        if "UNSEEN" in line:
            print(f"Testing line: {repr(line)}")
            print(f"Match: {RE_UNSEEN.search(line)}")
