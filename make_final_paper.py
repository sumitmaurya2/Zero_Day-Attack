#!/usr/bin/env python3
"""
ONE command that produces the final paper from real data.

    python make_final_paper.py --data MachineLearningCVE --emails "a@x.edu,b@x.edu,c@x.edu"

  1. runs run_experiments.py on your CIC-IDS2017 CSV folder (real measurements)
  2. runs build_paper.py on the results (fills every table, figure and sentence)
  3. writes Final_Paper.pdf and checks that no placeholder is left in it

Put run_experiments.py, build_paper.py and this file in the same folder.
Add --max-benign 100000 --max-per-family 15000 for a faster run on a laptop.
"""
import argparse, os, re, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with the CIC-IDS2017 MachineLearningCSV *.csv files")
ap.add_argument("--emails", required=True, help='three comma-separated emails')
ap.add_argument("--max-benign", default="200000"); ap.add_argument("--max-per-family", default="30000")
ap.add_argument("--out", default="Final_Paper.pdf")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable

subprocess.check_call([py, os.path.join(here, "run_experiments.py"), "--data", a.data, "--out", "results",
                       "--max-benign", a.max_benign, "--max-per-family", a.max_per_family])
subprocess.check_call([py, os.path.join(here, "build_paper.py"), "--results", "results", "--emails", a.emails, "--out", a.out])

from pypdf import PdfReader
r = PdfReader(a.out); text = " ".join(p.extract_text() or "" for p in r.pages)
bad = re.findall(r"\[(?:XX|n\]|0\.000|Describe|Which|State|Insert|Weakest|Realised|Fastest|Top five|Attack share|Hybrid cost|Weakest case|Best trade)", text)
print(f"\n{a.out}: {len(r.pages)} pages, placeholders left: {len(bad)}")
if bad or len(r.pages) > 19: sys.exit("Check the PDF before submitting.")
print("Done. Read the Results and Discussion section once and make sure the sentences match what you see in the tables.")
