import re
from pathlib import Path

path = Path("outputs/withholding_test.log")
RE_EXP     = re.compile(r"Experiment: Withholding m=(\d+) stations")
RE_SEEN    = re.compile(r"SEEN\s+mean MAE Tmax:\s*([\d.]+)")
RE_UNSEEN  = re.compile(r"UNSEEN\s+mean MAE Tmax:\s*([\d.]+)[^\d]+([\d.]+)")

cur_m = None
final = []

with open(path, encoding="utf-8") as f:
    for line in f:
        m = RE_EXP.search(line)
        if m:
            cur_m = int(m.group(1))
            print(f"Set cur_m={cur_m}")
            continue
            
        s = RE_SEEN.search(line)
        if s and cur_m is not None:
            if cur_m == 0:
                final.append({"m": 0})
            print(f"Matched SEEN for m={cur_m}")
            continue
            
        u = RE_UNSEEN.search(line)
        if u and cur_m is not None:
            print(f"Matched UNSEEN for m={cur_m}")
            final.append({"m": cur_m})
            continue

print("Final array:", len(final))
