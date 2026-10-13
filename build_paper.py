#!/usr/bin/env python3
"""
Builds the research paper PDF.

  python build_paper.py --results results --out AI_Zero_Day_Paper.pdf --emails "a@x.edu,b@x.edu,c@x.edu"

* With --results pointing at the folder written by run_experiments.py, every table, figure and
  data-driven sentence is filled from those measured results.
* Without --results it produces the DRAFT version: all text is final, and every value that must
  come from an experiment is a highlighted placeholder.
No result in this file is hard-coded.
"""
import argparse, json, os, re, tempfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                KeepTogether, PageBreak, Preformatted)

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None)
ap.add_argument("--out", default="AI_Zero_Day_Attack_Detection_Paper.pdf")
ap.add_argument("--emails", default=None, help="three comma-separated emails")
args = ap.parse_args()
RD = args.results
LIVE = RD is not None

# ----------------------------------------------------------------- styles
FONT, BOLD, ITAL = "Times-Roman", "Times-Bold", "Times-Italic"
body = ParagraphStyle("body", fontName=FONT, fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName=BOLD, fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName=BOLD, fontSize=11, leading=14, spaceBefore=8, spaceAfter=3, keepWithNext=1)
title = ParagraphStyle("title", fontName=BOLD, fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=8)
auth = ParagraphStyle("auth", fontName=FONT, fontSize=10.5, leading=13.5, alignment=TA_CENTER)
cap = ParagraphStyle("cap", fontName=ITAL, fontSize=9, leading=11.5, alignment=TA_CENTER, spaceAfter=9, spaceBefore=3)
tcap = ParagraphStyle("tcap", fontName=BOLD, fontSize=9, leading=11.5, alignment=TA_CENTER, spaceBefore=6, spaceAfter=3, keepWithNext=1)
cell = ParagraphStyle("cell", fontName=FONT, fontSize=8.3, leading=10, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName=BOLD)
bul = ParagraphStyle("bul", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3)
ref = ParagraphStyle("ref", fontName=FONT, fontSize=9.3, leading=11.8, leftIndent=22, firstLineIndent=-22,
                     spaceAfter=3, alignment=TA_LEFT)
HL = '<font backColor="#FFF3A3">%s</font>'

# ----------------------------------------------------------------- references (numbered by first citation)
REFS = {
 "buczak": 'A. L. Buczak and E. Guven, "A survey of data mining and machine learning methods for cyber security intrusion detection," <i>IEEE Commun. Surveys Tuts.</i>, vol. 18, no. 2, pp. 1153\u20131176, 2016.',
 "chandola": 'V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," <i>ACM Computing Surveys</i>, vol. 41, no. 3, art. 15, 2009.',
 "bilge": 'L. Bilge and T. Dumitras, "Before we knew it: An empirical study of zero-day attacks in the real world," in <i>Proc. ACM Conf. Computer and Communications Security (CCS)</i>, 2012, pp. 833\u2013844.',
 "iforest": 'F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in <i>Proc. IEEE Int. Conf. Data Mining (ICDM)</i>, 2008, pp. 413\u2013422.',
 "sakurada": 'M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in <i>Proc. MLSDA Workshop on Machine Learning for Sensory Data Analysis</i>, 2014.',
 "sommer": 'R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in <i>Proc. IEEE Symp. Security and Privacy</i>, 2010, pp. 305\u2013316.',
 "arp": 'D. Arp <i>et al.</i>, "Dos and don\'ts of machine learning in computer security," in <i>Proc. USENIX Security Symp.</i>, 2022.',
 "engelen": 'G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an intrusion detection dataset: The CICIDS2017 case study," in <i>Proc. IEEE Security and Privacy Workshops (SPW)</i>, 2021, pp. 7\u201312.',
 "breiman": 'L. Breiman, "Random forests," <i>Machine Learning</i>, vol. 45, no. 1, pp. 5\u201332, 2001.',
 "xgb": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785\u2013794.',
 "kitsune": 'Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, "Kitsune: An ensemble of autoencoders for online network intrusion detection," in <i>Proc. Network and Distributed System Security Symp. (NDSS)</i>, 2018.',
 "nslkdd": 'M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in <i>Proc. IEEE Symp. Computational Intelligence for Security and Defense Applications (CISDA)</i>, 2009.',
 "unsw": 'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in <i>Proc. Military Communications and Information Systems Conf. (MilCIS)</i>, 2015.',
 "cic17": 'I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in <i>Proc. Int. Conf. Information Systems Security and Privacy (ICISSP)</i>, 2018, pp. 108\u2013116.',
 "ring": 'M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, "A survey of network-based intrusion detection data sets," <i>Computers &amp; Security</i>, vol. 86, pp. 147\u2013167, 2019.',
 "cart": 'L. Breiman, J. Friedman, R. Olshen, and C. Stone, <i>Classification and Regression Trees</i>. Belmont, CA, USA: Wadsworth, 1984.',
 "friedman": 'J. H. Friedman, "Greedy function approximation: A gradient boosting machine," <i>Annals of Statistics</i>, vol. 29, no. 5, pp. 1189\u20131232, 2001.',
 "hinton": 'G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," <i>Science</i>, vol. 313, no. 5786, pp. 504\u2013507, 2006.',
 "sklearn": 'F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Machine Learning Research</i>, vol. 12, pp. 2825\u20132830, 2011.',
 "fawcett": 'T. Fawcett, "An introduction to ROC analysis," <i>Pattern Recognition Letters</i>, vol. 27, no. 8, pp. 861\u2013874, 2006.',
 "davis": 'J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in <i>Proc. Int. Conf. Machine Learning (ICML)</i>, 2006, pp. 233\u2013240.',
 "saito": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," <i>PLoS ONE</i>, vol. 10, no. 3, e0118432, 2015.',
 "goodfellow": 'I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2015.',
 "gama": 'J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," <i>ACM Computing Surveys</i>, vol. 46, no. 4, art. 44, 2014.',
 "lundberg": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017, pp. 4765\u20134774.',
 "mcmahan": 'B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in <i>Proc. Int. Conf. Artificial Intelligence and Statistics (AISTATS)</i>, 2017, pp. 1273\u20131282.',
 "kipf": 'T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2017.',
}
order = []
def cite(t):
    def sub(m):
        out = []
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order: order.append(k)
            out.append("[%d]" % (order.index(k) + 1))
        return ", ".join(out)
    return re.sub(r"\[\[([a-z0-9, ]+)\]\]", sub, t)

def ph(t="XX.X"): return HL % ("[" + t + "]")
def P(t, s=body): return Paragraph(cite(t), s)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)
def B(t): return Paragraph(cite(t), bul, bulletText="\u2022")

def table(data, widths, header=True):
    rows = [[Paragraph(cite(str(c)), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#555555")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3E8EF")))
    t.setStyle(TableStyle(st)); return t

def fig(path, w, caption):
    iw, ih = PI.open(path).size
    return KeepTogether([Image(path, width=w, height=w * ih / iw), P(caption, cap)])

def result_fig(fname, caption, w, ph_h):
    p = os.path.join(RD, fname) if LIVE else None
    if p and os.path.exists(p): return fig(p, w, caption)
    t = Table([[Paragraph("<i>Insert <b>%s</b> here</i><br/>(generated by run_experiments.py)" % fname,
                          ParagraphStyle("pp", fontName=FONT, fontSize=9, alignment=TA_CENTER, leading=12))]],
              colWidths=[5.6 * inch], rowHeights=[ph_h * inch])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.grey, None, (3, 2)),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBE0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return KeepTogether([t, P(caption, cap)])

def footer(c, d):
    c.saveState(); c.setFont(FONT, 9); c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page)); c.restoreState()

# ----------------------------------------------------------------- methodology diagrams
TMP = tempfile.mkdtemp()
def _box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1F4E8C", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def _arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
def _canvas(w, h):
    fig_, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); return fig_, ax

f_, ax = _canvas(9, 3.6)
_box(ax, .1, 1.4, 1.5, .9, "Network flow\nrecords\n(CIC-IDS2017)"); _box(ax, 1.95, 1.4, 1.5, .9, "Cleaning &\nfeature\ntransform")
_box(ax, 3.8, 2.3, 1.8, .85, "Supervised branch\nXGBoost classifier\n(benign + known attacks)", "#E6F4EA", "#2E7D32")
_box(ax, 3.8, .55, 1.8, .85, "Anomaly branch\nAutoencoder\n(benign traffic only)", "#FFF4E5", "#B26A00")
_box(ax, 5.95, 1.4, 1.35, .9, "Score fusion\nmax( p, z/(1+z) )"); _box(ax, 7.55, 1.4, 1.35, .9, "Threshold at\n1% benign FPR\n-> Alert / Pass", "#FDECEA", "#B3261E")
for a_ in [(1.6, 1.85, 1.95, 1.85), (3.45, 1.95, 3.8, 2.7), (3.45, 1.75, 3.8, 1.0), (5.6, 2.7, 5.95, 2.0), (5.6, 1.0, 5.95, 1.7), (7.3, 1.85, 7.55, 1.85)]: _arr(ax, *a_)
f_.savefig(f"{TMP}/f1.png", dpi=220, bbox_inches="tight"); plt.close()

f_, ax = _canvas(9, 1.7)
for i, s_ in enumerate(["Merge daily\nCSV files", "Strip names,\ndrop IDs, IPs,\ntimestamps", "Remove inf/NaN,\nduplicates,\nconstant cols", "Group labels\ninto 6 attack\nfamilies", "Cap class\nsizes\n(subsample)", "Signed log +\nstandardise\n(train only)"]):
    _box(ax, .05 + i * 1.5, .35, 1.3, .95, s_, fs=7.5)
    if i < 5: _arr(ax, 1.35 + i * 1.5, .82, 1.55 + i * 1.5, .82)
f_.savefig(f"{TMP}/f2.png", dpi=220, bbox_inches="tight"); plt.close()

f_, ax = _canvas(9, 2.6)
_box(ax, .1, 1, 1.6, 1, "Pick one attack\nfamily F\n(e.g. PortScan)"); _box(ax, 2, 1, 1.9, 1, "Remove ALL flows\nof F from the\ntraining data", "#FDECEA", "#B3261E")
_box(ax, 4.2, 1, 1.6, 1, "Train every\nmodel on the\nremainder"); _box(ax, 6.1, 1, 1.4, 1, "Set threshold on\nbenign validation\n(99th pct.)"); _box(ax, 7.75, 1, 1.2, 1, "Test on benign\ntest set + F\nonly")
for x1, x2 in [(1.7, 2), (3.9, 4.2), (5.8, 6.1), (7.5, 7.75)]: _arr(ax, x1, 1.5, x2, 1.5)
ax.text(4.5, .35, "Repeat for each family, then average over families", ha="center", fontsize=8.5, style="italic")
f_.savefig(f"{TMP}/f3.png", dpi=220, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------- load results (live mode)
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Isolation Forest", "Autoencoder", "Proposed Hybrid"]
FAMS = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]
if LIVE:
    zd = pd.read_csv(f"{RD}/zero_day_results.csv"); kn = pd.read_csv(f"{RD}/known_attack_results.csv")
    comp = pd.read_csv(f"{RD}/dataset_composition.csv"); env = json.load(open(f"{RD}/environment.json"))
    fi = pd.read_csv(f"{RD}/feature_importance.csv", index_col=0)["importance"]
    FAMS = [f for f in FAMS if f in set(zd.heldout)]
    zs = zd.groupby("model")[["precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "latency_ms_per_1k"]].mean()
    ks = kn.set_index("model")
    for d_ in (zs, ks): d_ = d_.reindex(MODELS)
    zs = zs.reindex(MODELS)
else:
    # Draft mode: dummy frames only so the table code can run; V() never shows these values.
    _cols = ["precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "latency_ms_per_1k"]
    zs = pd.DataFrame(0.0, index=MODELS, columns=_cols); ks = zs.copy()
    zd = pd.DataFrame({"model": [m for m in MODELS for _ in FAMS], "heldout": FAMS * len(MODELS), "recall": 0.0})
    comp = env = fi = None
def f3(x): return "%.3f" % x
def pc(x): return "%.1f" % (100 * x)
def V(draft="0.000", live=None): return live if LIVE else ph(draft)

# ----------------------------------------------------------------- data-driven prose
if LIVE:
    best = zs["f1"].idxmax(); hy, xg = zs.loc["Proposed Hybrid"], zs.loc["XGBoost"]
    diff = hy.recall - xg.recall
    famrec = zd.groupby("heldout")["recall"].mean(); hard, easy = famrec.idxmin(), famrec.idxmax()
    drop = (ks["recall"] - zs["recall"]); dmax, dmin = drop.idxmax(), drop.idxmin()
    zd["prev"] = (zd.TP + zd.FN) / (zd.TP + zd.FN + zd.TN + zd.FP); kn["prev"] = (kn.TP + kn.FN) / (kn.TP + kn.FN + kn.TN + kn.FP)
    if diff > 0.01: verdict = "The hybrid detector raised the mean detection rate on unseen families by %.1f percentage points over its XGBoost component (%s%% versus %s%%)." % (100 * diff, pc(hy.recall), pc(xg.recall))
    elif diff < -0.01: verdict = "The hybrid detector did not help: its mean detection rate on unseen families was %.1f percentage points lower than XGBoost alone (%s%% versus %s%%)." % (-100 * diff, pc(hy.recall), pc(xg.recall))
    else: verdict = "The hybrid detector was within one percentage point of XGBoost alone on unseen families (%s%% versus %s%%), so the autoencoder branch added little in this experiment." % (pc(hy.recall), pc(xg.recall))
    lat_cost = hy.latency_ms_per_1k - xg.latency_ms_per_1k
    T = dict(
      abs_nums=("On unseen families the models reached mean detection rates between %s%% and %s%%; the hybrid detector scored %s%% with a false-positive rate of %s%% and its scoring cost differed from XGBoost by %.1f ms per 1,000 flows. Detection was weakest for the %s family (mean %s%% across models). "
                % (pc(zs.recall.min()), pc(zs.recall.max()), pc(hy.recall), pc(hy.fpr), lat_cost, hard, pc(famrec.min()))),
      cm=("Summed over all held-out folds, the best model by mean F1 (%s) produced %s true positives, %s false negatives, %s false positives and %s true negatives." %
          (best, *(f"{int(zd[zd.model == best][c].sum()):,}" for c in ["TP", "FN", "FP", "TN"]))),
      cmp=("%s achieved the highest mean F1 on unseen families (%s), and %s the lowest (%s). Relative to the known-attack scenario, mean detection fell most for %s (by %.1f percentage points) and least for %s (by %.1f). %s"
           % (best, f3(zs.f1.max()), zs.f1.idxmin(), f3(zs.f1.min()), dmax, 100 * drop.max(), dmin, 100 * drop.min(), verdict)),
      fpfn=("The realised false-positive rate ranged from %s%% to %s%% across all models and folds, against the 1%% target set on validation data. The highest false-negative rate belonged to the %s family (mean detection %s%%), while %s was easiest (mean detection %s%%). Cases like this usually mean the held-out behaviour sits close to benign traffic in feature space, so the learned boundary never separates it."
            % (pc(zd.fpr.min()), pc(zd.fpr.max()), hard, pc(famrec.min()), easy, pc(famrec.max()))),
      lat=("Scoring was fastest for %s (%.1f ms per 1,000 flows) and slowest for %s (%.1f ms). The hybrid detector needed %.1f ms per 1,000 flows, which is %.1f ms %s than XGBoost alone." %
           (zs.latency_ms_per_1k.idxmin(), zs.latency_ms_per_1k.min(), zs.latency_ms_per_1k.idxmax(), zs.latency_ms_per_1k.max(), hy.latency_ms_per_1k, abs(lat_cost), "more" if lat_cost >= 0 else "less")),
      feat=("The five most important features in the Random Forest were %s. Features of this kind describe packet-size, timing or flag behaviour of a flow, which is why they separate attack tools that behave mechanically from interactive human traffic." % ", ".join(f"<i>{n}</i>" for n in fi.index[:5])),
      q1=("Held-out families were detected at mean rates between %s%% and %s%% depending on the model; %s was easiest and %s hardest." % (pc(zs.recall.min()), pc(zs.recall.max()), easy, hard)),
      q2=("At the shared 1%% false-positive budget, %s gave the best mean F1 (%s) and %s the best mean detection rate (%s%%)." % (best, f3(zs.f1.max()), zs.recall.idxmax(), pc(zs.recall.max()))),
      q3=("Attack flows made up %s%% to %s%% of the zero-day test sets and %s%% of the known-attack test set. Precision depends directly on this ratio, so precision and F1 from this dataset will not carry over to a network where attacks are rarer; PR-AUC and the false-positive rate are the safer numbers to quote." % (pc(zd.prev.min()), pc(zd.prev.max()), pc(kn.prev.mean()))),
      q5=("The hybrid's scoring cost was %.1f ms per 1,000 flows against %.1f ms for XGBoost. %s" % (hy.latency_ms_per_1k, xg.latency_ms_per_1k, verdict)),
      q6=("The weakest case was the %s family (mean detection %s%%). The most likely reason is overlap with benign behaviour in the flow features, together with the fact that no similar attack was present during training." % (hard, pc(famrec.min()))),
      concl=("Detection rates on unseen families ranged from %s%% to %s%%, compared with %s%% to %s%% when the same families were seen in training. %s Scoring the hybrid cost %.1f ms per 1,000 flows more than XGBoost alone." %
             (pc(zs.recall.min()), pc(zs.recall.max()), pc(ks.recall.min()), pc(ks.recall.max()), verdict, lat_cost)),
    )
else:
    T = dict(abs_nums="On unseen families the models reached mean detection rates between " + ph() + "% and " + ph() + "%; the hybrid detector scored " + ph() + "% with a false-positive rate of " + ph("X.X") + "% and its scoring cost differed from XGBoost by " + ph("X.X") + " ms per 1,000 flows. " + ph("Weakest family sentence") + " ",
             cm=ph("Describe the confusion matrix: attacks missed, benign flows flagged"), cmp=ph("Which model is best on unseen families, how much each drops from Table IV to Table V, whether the hybrid helps"),
             fpfn=ph("Realised FPR range, family with the highest FNR and why"), lat=ph("Fastest and slowest model, hybrid cost versus XGBoost"),
             feat=ph("Top five features and what they suggest"), q1=ph("Which families were detected, which were hard"), q2=ph("Best trade-off model"),
             q3=ph("Attack share of the test sets and what it means for precision"), q5=ph("Hybrid cost versus gain"), q6=ph("Weakest case and likely reason"),
             concl="Detection rates on unseen families ranged from " + ph() + "% to " + ph() + "%, compared with " + ph() + "% to " + ph() + "% when the same families were seen in training. " + ph("State whether the hybrid helped, by how much, and at what latency cost.") + " ")

# =================================================================== DOCUMENT
S = []
emails = (args.emails or "").split(",") if args.emails else None
S += [P("AI-Based Zero-Day Attack Detection Using Machine Learning", title),
      P("<i>A Leave-One-Attack-Family-Out Evaluation with a Hybrid Supervised\u2013Anomaly Detector</i>", ParagraphStyle("st", parent=auth, fontSize=11, spaceAfter=10)),
      P("<b>Sumit Maurya, Sumit Vishwakarma, Sunil Gupta</b>", auth),
      P("Department of Computer Science &amp; Engineering, B.Tech (Final Year)", auth),
      P("R.R. Institute of Modern Technology, Lucknow", auth),
      P(" &nbsp;|&nbsp; ".join(e.strip() for e in emails) if emails else HL % "[email 1] &nbsp;|&nbsp; [email 2] &nbsp;|&nbsp; [email 3]", auth),
      Spacer(1, 10)]

S += [H1("ABSTRACT"),
      P("A zero-day attack abuses a weakness that nobody has documented yet, so a signature-based intrusion detection system has nothing to match it against. "
        "Machine learning offers another route: learn what normal and malicious traffic look like instead of memorising known patterns. The catch is that many "
        "published results come from random train/test splits, which quietly put every attack type into the training data and make detectors look better than they "
        "would be on genuinely new behaviour. In this paper we ask how well common machine-learning detectors cope with attack families they have never seen. Using "
        "flow-level records from the CIC-IDS2017 dataset, we compare Logistic Regression, a Decision Tree, Random Forest, XGBoost, Isolation Forest and an autoencoder "
        "with a hybrid detector that fuses a gradient-boosted classifier and a benign-only autoencoder. Generalisation is tested with a leave-one-attack-family-out "
        "protocol, and every model is calibrated to the same 1% false-positive budget so the comparison is fair. " + T["abs_nums"] +
        "The held-out setting is an experimental approximation of a zero-day scenario and does not prove detection of real-world zero-day attacks."),
      P("<b>Keywords&mdash;</b><i>zero-day attack, intrusion detection, machine learning, anomaly detection, CIC-IDS2017, generalisation.</i>")]

# ---------------------------------------------------------------- 1 Introduction
S += [H1("1. INTRODUCTION"), H2("1.1 Background"),
      P("Almost everything we do now, from banking to attendance records, depends on networked systems, and those systems are probed constantly. An intrusion detection "
        "system (IDS) sits on a host or a network link and tries to separate hostile activity from ordinary use. Two broad families exist. Misuse (signature) detection "
        "compares traffic with rules written for attacks that are already known. Anomaly detection builds a picture of normal behaviour and raises an alert when something "
        "departs from it [[buczak,chandola]]. Signature systems are precise on what they know and silent on everything else, which is the root of the zero-day problem."),
      P("The practical stakes are easy to see. A university, a hospital or a small company rarely has a security team watching dashboards all day. If the detector "
        "misses a new attack, nobody notices; if it fires too often, people stop reading its alerts. Any approach that claims to help has to address both failures together, "
        "and that is why this paper measures detection and false alarms side by side rather than reporting accuracy alone."),
      H2("1.2 Zero-Day Attacks"),
      P("A zero-day attack exploits a vulnerability that the vendor and the defenders do not yet know about, so no patch and no signature exists on the day of the attack. "
        "Bilge and Dumitras found that such exploits can stay in use for a long time before they are discovered [[bilge]], which is exactly the window in which signature "
        "engines are blind. A detector that is useful in this window must recognise <i>behaviour</i> rather than a fingerprint: a port scan looks like a port scan, and a "
        "flood looks like a flood, even when the specific tool or exploit has never been catalogued."),
      H2("1.3 Machine Learning in Cybersecurity"),
      P("Supervised classifiers learn the boundary between benign and malicious flows from labelled examples. Unsupervised methods such as Isolation Forest [[iforest]] and "
        "autoencoders [[sakurada]] only need benign data and flag whatever looks unusual. Each has an obvious weakness. A supervised model may simply memorise the attacks it "
        "was shown, while an anomaly model can fire on harmless but rare traffic and bury analysts in false alarms. Sommer and Paxson made this point early: intrusion "
        "detection is a harder fit for machine learning than most other applications, mainly because of the cost of errors and the difficulty of evaluation [[sommer]]."),
      H2("1.4 Research Gap"),
      P("Reported accuracies on intrusion datasets are often very high, but the evaluation is usually a random split in which each attack type appears on both sides. "
        "That tests whether a model has seen an attack before, not whether it can handle one it has not. Arp et al. list this kind of sampling bias among the common pitfalls "
        "of machine learning in security research [[arp]]. Public datasets also have their own flaws; CIC-IDS2017 in particular has documented labelling and flow-construction "
        "problems [[engelen]]. What is missing is a careful, reproducible comparison of common models under a held-out-attack protocol, at a fixed false-positive rate, "
        "that also reports computational cost."),
      H2("1.5 Contributions"),
      B("A leave-one-attack-family-out evaluation protocol in which all models share one false-positive budget (1%)."),
      B("A hybrid detector that fuses an XGBoost classifier with a benign-only autoencoder by taking the larger of the two normalised scores."),
      B("A side-by-side study of six baseline models using precision, recall, F1, false-positive and false-negative rates, ROC-AUC, PR-AUC and scoring latency."),
      B("An analysis of where each model fails, and a fully scripted pipeline so every number in the paper can be regenerated."),
      H2("1.6 Organisation of the Paper"),
      P("Section 2 reviews related work. Sections 3 and 4 state the problem formally and list the objectives. Section 5 describes the dataset, preprocessing, models, "
        "evaluation protocol and metrics, and Section 6 gives the experimental setup. Section 7 reports and discusses the results. Sections 8 to 10 cover limitations, "
        "future scope and conclusions.")]

# ---------------------------------------------------------------- 2 Related work
S += [H1("2. RELATED WORK / LITERATURE REVIEW"), H2("2.1 Signature-Based and Anomaly-Based Detection"),
      P("Buczak and Guven surveyed data-mining and machine-learning techniques for intrusion detection and described the trade-offs between misuse and anomaly approaches "
        "[[buczak]]. Chandola et al. gave the general taxonomy of anomaly detection on which many network detectors are built [[chandola]], separating point anomalies, "
        "contextual anomalies and collective anomalies. For flow-level intrusion data, most attacks of interest show up as point anomalies in a feature space of packet "
        "counts, sizes and timings, which is the view we adopt here."),
      H2("2.2 Supervised Learning for Intrusion Detection"),
      P("Tree-based supervised methods are the workhorses of tabular security data. Decision trees [[cart]] give transparent rules, Random Forest [[breiman]] averages many "
        "decorrelated trees to cut variance, and gradient boosting [[friedman]], in its engineered form XGBoost [[xgb]], builds trees sequentially to correct earlier errors. "
        "These methods train quickly, tolerate unscaled features and often give strong results when train and test traffic come from the same distribution. Their "
        "weakness is the thing this paper studies: they learn the attacks they are shown."),
      H2("2.3 Unsupervised and Deep Anomaly Detection"),
      P("Isolation Forest [[iforest]] isolates points by random axis-parallel splits; anomalies need fewer splits to be isolated, so short average path lengths mean high "
        "anomaly scores. Autoencoders compress the input through a narrow hidden layer and reconstruct it; a model trained only on normal data reconstructs normal "
        "inputs well and unfamiliar inputs poorly [[hinton,sakurada]]. Mirsky et al. extended this idea to an online ensemble of autoencoders for network intrusion "
        "detection in the Kitsune system [[kitsune]]. Anomaly methods need no attack labels at all, but they pay for that freedom with more false alarms."),
      H2("2.4 Datasets"),
      P("Datasets matter as much as models. Ring et al. compared the public network-intrusion datasets and the properties that make each more or less useful [[ring]]. "
        "KDD Cup 99 was analysed and criticised by Tavallaee et al., who produced NSL-KDD [[nslkdd]]; UNSW-NB15 [[unsw]] and CIC-IDS2017 [[cic17]] were later built to "
        "reflect more modern traffic and attacks. Engelen et al. then re-examined CIC-IDS2017 and reported labelling and flow-construction problems in the original release "
        "[[engelen]], which is why we clean the data conservatively and keep our claims modest."),
      H2("2.5 Evaluation Practice"),
      P("Sommer and Paxson [[sommer]] and Arp et al. [[arp]] argue that evaluation design, not model choice, is where most security-ML studies go wrong. Two further points "
        "shape our metrics. First, ROC analysis [[fawcett]] is threshold-independent but can look optimistic when attacks are rare, and precision\u2013recall analysis is "
        "more informative under class imbalance [[davis,saito]]. Second, a detector is only useful at a stated operating point, so we fix the false-positive budget "
        "rather than letting each model pick its own threshold."),
      H2("2.6 Summary and Position of This Study"),
      P("Table I summarises how these works relate to ours. In short, the algorithms are well understood and the datasets are available; what is thinner is a "
        "reproducible, fixed-operating-point comparison of common models on <i>held-out attack families</i> with cost reported alongside accuracy. That is the gap "
        "this study targets."),
      P("<b>TABLE I.</b> Related work and the position of this study.", tcap),
      table([["Study", "Year", "Approach", "Data / setting", "Contribution", "Gap relevant to this work"],
             ["Sommer &amp; Paxson [[sommer]]", "2010", "Critical analysis", "Network IDS practice", "Explains why ML is hard to apply to intrusion detection", "No experimental comparison of models"],
             ["Sakurada &amp; Yairi [[sakurada]]", "2014", "Autoencoder anomaly detection", "General (non-network) data", "Reconstruction error as anomaly score", "Not evaluated on network flows"],
             ["Mirsky et al. [[kitsune]]", "2018", "Ensemble of autoencoders (Kitsune)", "Own captured attack traces", "Online, unsupervised NIDS", "Benign-only; no supervised comparison on a public flow set"],
             ["Ring et al. [[ring]]", "2019", "Dataset survey", "Public NIDS datasets", "Compares dataset properties", "Dataset study, not a detector"],
             ["Engelen et al. [[engelen]]", "2021", "Dataset audit", "CIC-IDS2017", "Documents labelling and flow errors", "Dataset study, not a detector"],
             ["Arp et al. [[arp]]", "2022", "Literature analysis", "ML-in-security papers", "Catalogues evaluation pitfalls", "Guidance, no new detector"],
             ["<b>This study</b>", "2026", "Hybrid XGBoost + autoencoder", "CIC-IDS2017, leave-one-family-out", "Fair, fixed-FPR zero-day style comparison", "Evaluated in Section 7"]],
            [0.95 * inch, 0.4 * inch, 1.05 * inch, 1.0 * inch, 1.4 * inch, 1.35 * inch])]

# ---------------------------------------------------------------- 3 Problem statement
S += [Spacer(1, 4), H1("3. PROBLEM STATEMENT"), H2("3.1 Informal Statement"),
      P("Traditional defences depend on known attack signatures and therefore struggle with previously unseen attacks. This study investigates whether machine-learning-based "
        "traffic analysis can detect attack behaviour that was absent from the training data while keeping the false-positive rate at an operationally acceptable level "
        "(here, 1% of benign flows). The question is deliberately framed as generalisation to unseen attack <i>families</i>, because a model that only recognises variants "
        "of what it has already seen has not solved the zero-day problem."),
      H2("3.2 Formal Statement"),
      P("Let each network flow be a feature vector <i>x</i> in R<super>d</super> with label <i>y</i> in {0, 1}, where 0 is benign and 1 is an attack, and let each attack "
        "flow additionally belong to a family <i>f</i> in F = {DoS, DDoS, PortScan, BruteForce, WebAttack, Bot}. A detector is a scoring function "
        "<i>s</i>(<i>x</i>) with an alert threshold \u03c4: it raises an alert when <i>s</i>(<i>x</i>) &gt; \u03c4. For a held-out family <i>f*</i>, the model is "
        "trained on benign flows and on attacks from F minus {<i>f*</i>}. The threshold \u03c4 is chosen so that the false-positive rate on benign validation data is "
        "at most \u03b1 = 1%. The quantity of interest is the detection rate on the unseen family, TPR(<i>f*</i>) = P(<i>s</i>(<i>x</i>) &gt; \u03c4 | family = <i>f*</i>), "
        "averaged over all choices of <i>f*</i>. A better detector is one with a higher average TPR(<i>f*</i>) at the same \u03b1."),
      H2("3.3 Assumptions and Scope"),
      B("Detection is at the level of single flows, using the flow features provided in CIC-IDS2017; payload inspection is out of scope."),
      B("The benign training data are assumed to be clean, i.e. free of undetected attacks, which is the standard assumption for anomaly detectors."),
      B("A \u201czero-day\u201d here means an attack <i>family</i> absent from training. It is a controlled stand-in, not a claim about attacks that exploit unknown software flaws."),
      H1("4. RESEARCH OBJECTIVES"),
      B("O1. Build a machine-learning framework for detecting previously unseen cyberattacks."),
      B("O2. Clean and analyse flow-level network data suitable for this purpose."),
      B("O3. Evaluate several models with standard metrics."),
      B("O4. Compare the proposed hybrid detector with baseline methods."),
      B("O5. Analyse false positives, false negatives and generalisation to unseen attack families."),
      B("O6. Assess whether the approach is cheap enough for real-time or near-real-time use.")]

# ---------------------------------------------------------------- 5 Methodology
S += [H1("5. PROPOSED METHODOLOGY"), H2("5.1 System Architecture"),
      P("The detector has two branches that look at the same preprocessed flow. The supervised branch is an XGBoost classifier trained on benign flows and the attack "
        "families that are allowed in training. The anomaly branch is an autoencoder trained on benign flows only, so it has no idea what any attack looks like. The "
        "branches are fused at the score level and the result is compared with a threshold chosen on benign validation data (Fig. 1). The design idea is that the two "
        "branches fail differently: the classifier is sharp on attacks it has seen and blind to new ones, while the autoencoder is blunt but does not care whether an "
        "attack is new."),
      fig(f"{TMP}/f1.png", 6.0 * inch, "Fig. 1. Overall architecture of the proposed hybrid zero-day detection system."),
      H2("5.2 Dataset"),
      P("We use CIC-IDS2017 [[cic17]], collected by the Canadian Institute for Cybersecurity over five working days in July 2017. It contains benign traffic together with "
        "brute-force (FTP and SSH), DoS, Heartbleed, web attacks, infiltration, botnet, DDoS and port-scan activity, summarised into more than 70 statistical flow features "
        "by CICFlowMeter. We group the labels into six attack families: DoS (Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed), DDoS, PortScan, BruteForce "
        "(FTP-Patator, SSH-Patator), WebAttack (brute force, XSS, SQL injection) and Bot. Infiltration is excluded because it contains only a few dozen flows, too few to "
        "hold out meaningfully. Table II reports the composition actually used after cleaning and subsampling."),
      P("<b>TABLE II.</b> Dataset composition after cleaning and subsampling.", tcap)]
if LIVE:
    rows = [["Class / family", "Flows used", "Train", "Validation", "Test"]] + [[r.cls, f"{r.total:,}", f"{r.train:,}", f"{r.val:,}" if r.val else "-", f"{r.test:,}"] for r in comp.itertuples()]
    rows.append(["Total (features: %d)" % env["n_features"], f"{comp.total.sum():,}", f"{comp.train.sum():,}", f"{comp.val.sum():,}", f"{comp.test.sum():,}"])
else:
    rows = [["Class / family", "Flows used", "Train", "Validation", "Test"]] + [[c] + [ph("n")] * 4 for c in ["BENIGN"] + FAMS] + [["Total / feature count"] + [ph("n")] * 4]
S += [table(rows, [1.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]), Spacer(1, 6),
      P("Attack families are split 70/30 into training and test portions for the known-attack scenario; in the zero-day scenario the held-out family is removed from "
        "training and <i>all</i> of its flows are used for testing. Benign flows are split 60/20/20 into training, validation and test sets."),
      H2("5.3 Data Preprocessing"),
      P("Fig. 2 shows the pipeline. We strip whitespace from column names, then drop the flow ID, source and destination IP, timestamp and source port. These identifiers "
        "are artefacts of how the lab traffic was generated, and leaving them in lets a model succeed by recognising a machine instead of a behaviour. Infinite and missing "
        "values are removed, as are exact duplicates and constant columns. Class sizes are capped so that experiments finish on ordinary hardware. Because flow features "
        "span many orders of magnitude and a few are negative, we apply a sign-preserving log transform, sign(x)\u00b7log(1+|x|), and then standardise using the mean "
        "and variance of the <i>training</i> split only, so that no test information leaks into the scaler."),
      fig(f"{TMP}/f2.png", 6.2 * inch, "Fig. 2. Data preprocessing and feature-engineering pipeline."),
      H2("5.4 Feature Engineering"),
      P("We keep the native CICFlowMeter features (flow duration, packet and byte counts, inter-arrival time statistics, TCP flag counts, window sizes and similar) rather "
        "than inventing new ones, so that differences between models come from the models and not from hand-built features. The only transformation is the log scaling "
        "above. Feature relevance is examined after training using Random Forest importance scores (Section 7.6)."),
      H2("5.5 Model Selection"),
      P("We use a small, defensible set rather than a long list tested without analysis. Table III summarises the role of each model, and the paragraphs after it describe "
        "how each one works and why it is included."),
      P("<b>TABLE III.</b> Models and their roles.", tcap),
      table([["Model", "Type", "Trained on", "Role in this study"],
             ["Logistic Regression", "Linear classifier", "Benign + known attacks", "Simple baseline"],
             ["Decision Tree", "Tree classifier [[cart]]", "Benign + known attacks", "Interpretable baseline"],
             ["Random Forest", "Tree ensemble [[breiman]]", "Benign + known attacks", "Strong tree-based baseline"],
             ["XGBoost", "Gradient boosting [[xgb]]", "Benign + known attacks", "High-performance supervised model"],
             ["Isolation Forest", "Unsupervised [[iforest]]", "Benign only", "Anomaly-detection baseline"],
             ["Autoencoder", "Neural, unsupervised [[sakurada]]", "Benign only", "Deep anomaly-detection baseline"],
             ["Proposed Hybrid", "XGBoost + autoencoder", "Benign + known attacks (classifier); benign (autoencoder)", "Proposed model"]],
            [1.2 * inch, 1.55 * inch, 1.9 * inch, 1.55 * inch]),
      Spacer(1, 6),
      P("<b>Logistic Regression</b> fits a linear boundary and outputs a probability; it is fast, easy to explain and a floor that more complex models should clear. "
        "<b>The Decision Tree</b> partitions feature space with threshold rules, giving human-readable decisions but a tendency to overfit unless depth is limited, so we cap "
        "depth and leaf size. <b>Random Forest</b> averages many randomised trees, which reduces variance and usually generalises better than a single tree. "
        "<b>XGBoost</b> adds trees one at a time to correct the current model's mistakes and is a frequent top performer on tabular data. <b>Isolation Forest</b> never sees "
        "attack labels; it isolates unusual points quickly and so assigns them higher anomaly scores. The <b>autoencoder</b> is a multilayer perceptron with a narrow middle "
        "layer trained to rebuild benign flows, so a large reconstruction error signals unfamiliar behaviour."),
      P("<b>The proposed hybrid.</b> If <i>p</i> is the XGBoost attack probability and <i>e</i> is the autoencoder reconstruction error, we compute z = e / q<sub>99</sub>, "
        "where q<sub>99</sub> is the 99th percentile of <i>e</i> on benign validation data, and use the fused score s = max(<i>p</i>, z/(1+z)). The mapping z/(1+z) puts "
        "the anomaly score on a 0\u20131 scale comparable with a probability, with 0.5 at the 99th percentile of normal behaviour. The rule is intentionally simple: the flow "
        "is suspicious if <i>either</i> branch is suspicious, so the autoencoder can catch what the classifier was never trained to recognise, while the classifier keeps "
        "precision high on known behaviour."),
      H2("5.6 Zero-Day Evaluation Strategy"),
      P("For the zero-day scenario we run a leave-one-attack-family-out loop (Fig. 3): one family is removed from training entirely, all models are trained on what "
        "remains, and testing uses the benign test set together with every flow of the held-out family. For each model the alert threshold is the 99th percentile of its "
        "score on benign validation data, which fixes the false-positive budget near 1% and makes detection rates directly comparable. The known-attack scenario, with all "
        "families present in training, is run alongside as an upper reference. We stress that the zero-day scenario is an <i>experimental approximation</i> of a real "
        "zero-day situation, not an actual zero-day."),
      fig(f"{TMP}/f3.png", 6.2 * inch, "Fig. 3. Model training and zero-day (held-out family) evaluation workflow."),
      H2("5.7 Evaluation Metrics"),
      P("Let TP, FP, TN and FN be the counts of true positives, false positives, true negatives and false negatives, with \u201cattack\u201d as the positive class. Table IV "
        "defines the metrics we report. Accuracy is deliberately not used as a headline figure because it is dominated by the benign majority."),
      P("<b>TABLE IV.</b> Evaluation metrics.", tcap),
      table([["Metric", "Definition", "Why it is reported"],
             ["Precision", "TP / (TP + FP)", "Share of alerts that are real attacks"],
             ["Recall (detection rate)", "TP / (TP + FN)", "Share of attacks that are caught"],
             ["F1-score", "2 \u00b7 Precision \u00b7 Recall / (Precision + Recall)", "Balance of the two above"],
             ["False-positive rate (FPR)", "FP / (FP + TN)", "How often benign traffic is flagged"],
             ["False-negative rate (FNR)", "FN / (FN + TP) = 1 \u2212 Recall", "How often attacks are missed"],
             ["ROC-AUC", "Area under TPR-vs-FPR curve [[fawcett]]", "Threshold-independent ranking quality"],
             ["PR-AUC", "Area under precision-recall curve [[davis,saito]]", "Ranking quality under class imbalance"],
             ["Latency", "Scoring time per 1,000 flows", "Feasibility for real-time use"]],
            [1.6 * inch, 2.7 * inch, 1.9 * inch]),
      Spacer(1, 6),
      H2("5.8 Algorithm Summary"),
      P("The complete protocol is summarised below; it is exactly what the accompanying script executes."),
      Table([[Preformatted(
"""for each scenario in {known-attack, zero-day}:
    for each held-out family f* (zero-day only):
        train_set  = benign_train + attacks_train(families except f*)
        test_set   = benign_test  + (all flows of f*  |  attack_test)
        fit scaler on train_set; transform train/validation/test
        for each model m:
            fit m on train_set       (anomaly models: benign_train only)
            tau = 99th percentile of m.score(benign_validation)
            alert = m.score(test_set) > tau
            record precision, recall, F1, FPR, FNR, ROC-AUC, PR-AUC, latency
report mean over held-out families""", ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10))]], colWidths=[6.2 * inch],
            style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.grey), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))]

# ---------------------------------------------------------------- 6 Setup
if LIVE:
    sw = "Python %s; scikit-learn %s; XGBoost %s; NumPy %s; pandas %s" % (env["python"], env["sklearn"], env["xgboost"], env["numpy"], env["pandas"])
    hw = "%s; %s logical cores; RAM %s GB; GPU %s" % (env["cpu"], env["cores"], env["ram_gb"], env["gpu"])
else:
    sw, hw = ph("Python x.x, scikit-learn x.x, XGBoost x.x"), ph("CPU model, RAM, GPU or none")
S += [H1("6. EXPERIMENTAL SETUP"), H2("6.1 Software and Hardware"),
      P("All models were implemented in Python with scikit-learn [[sklearn]] and XGBoost [[xgb]]. Every random component uses seed 42. The complete pipeline is a single "
        "script so the work can be reproduced end to end. Table V lists the environment and the settings, which were fixed in advance and not tuned on test data."),
      P("<b>TABLE V.</b> Model hyperparameters and environment.", tcap),
      table([["Item", "Setting"],
             ["Logistic Regression", "max_iter = 1000, class_weight = balanced"],
             ["Decision Tree", "max_depth = 12, min_samples_leaf = 5, class_weight = balanced"],
             ["Random Forest", "200 trees, class_weight = balanced_subsample"],
             ["XGBoost", "300 estimators, max_depth = 6, learning rate 0.1, subsample 0.8, colsample_bytree 0.8"],
             ["Isolation Forest", "200 trees, trained on benign flows only"],
             ["Autoencoder", "MLP 32-16-32, ReLU, max 40 epochs, early stopping, benign only (up to 100,000 flows)"],
             ["Proposed hybrid", "XGBoost + autoencoder, fused score s = max(p, z/(1+z))"],
             ["Threshold rule", "99th percentile of score on benign validation set (all models)"],
             ["Software", sw], ["Hardware", hw]], [1.6 * inch, 4.6 * inch]),
      H2("6.2 Design Choices"),
      P("Three choices deserve a short justification. First, thresholds are set on a benign <i>validation</i> split that is never used for training or testing, so that the "
        "operating point is chosen without peeking at the test set. Second, the scaler and the autoencoder reference quantile are fitted on training or validation data only. "
        "Third, class weights are used in the supervised baselines so that the benign majority does not drown out the attack classes, but no oversampling is applied, "
        "so the class ratios in the test data remain realistic for the dataset."),
      H2("6.3 Reproducibility"),
      P("The script writes every table and figure used in this paper to a results folder, together with a record of the software versions and hardware. The paper itself "
        "is built from that folder by a second script, so no number is typed by hand. To reproduce the study, download the CIC-IDS2017 \u201cMachineLearningCVE\u201d "
        "CSV files, run the experiment script with the data folder as its argument, and then run the paper script with the results folder (Appendix B).")]

# ---------------------------------------------------------------- 7 Results
S += [H1("7. RESULTS AND DISCUSSION")]
if not LIVE:
    S += [P("<i>Draft status: every highlighted cell is filled automatically from the experiment output when the paper script is run with --results. "
            "Nothing in this draft is an invented result.</i>", ParagraphStyle("n", parent=body, textColor=colors.HexColor("#8A5A00")))]
S += [H2("7.1 Performance Metrics"),
      P("Table VI gives results when every attack family is present in training, which is the setting most published work uses. It serves as the upper reference for "
        "Table VII, where one family is hidden at a time. Because attacks are rare compared with benign flows, we report PR-AUC next to ROC-AUC [[saito]]."),
      P("<b>TABLE VI.</b> Known-attack scenario (all families seen during training), threshold at ~1% benign FPR.", tcap)]
hdr = ["Model", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "PR-AUC"]
S += [table([hdr] + [[m] + [V("0.000", f3(ks.loc[m][c])) for c in ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]] for m in MODELS], [1.6 * inch] + [0.73 * inch] * 6),
      P("<b>TABLE VII.</b> Zero-day scenario: detection rate (recall) on each held-out family, at ~1% benign FPR.", tcap)]
piv = zd.pivot(index="model", columns="heldout", values="recall")
S += [table([["Model"] + FAMS + ["Mean"]] + [[m] + [V("0.000", f3(piv.loc[m][f])) for f in FAMS] + [V("0.000", f3(zs.loc[m].recall))] for m in MODELS],
            [1.2 * inch] + [0.76 * inch] * (len(FAMS) + 1)),
      H2("7.2 Confusion Matrix"),
      P("Fig. 4 shows the confusion matrix of the best-performing model, summed over all held-out folds. " + T["cm"]),
      result_fig("fig_confusion_matrix.png", "Fig. 4. Confusion matrix of the best-performing model (summed over zero-day folds).", 3.2 * inch, 2.2),
      H2("7.3 Model Comparison"),
      P("<b>TABLE VIII.</b> Zero-day scenario, mean over the held-out families.", tcap),
      table([["Model", "Precision", "Recall", "F1", "FPR", "FNR", "ROC-AUC", "PR-AUC"]] +
            [[m] + [V("0.000", f3(zs.loc[m][c])) for c in ["precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]] for m in MODELS], [1.2 * inch] + [0.76 * inch] * 7),
      Spacer(1, 6), P(T["cmp"]),
      result_fig("fig_model_comparison.png", "Fig. 5. Detection rate per held-out family for all models.", 5.8 * inch, 2.4),
      result_fig("fig_roc_zero_day.png", "Fig. 6. ROC curves for each held-out family.", 6.0 * inch, 2.6),
      H2("7.4 False-Positive / False-Negative Analysis"),
      P("Because every threshold is tied to a 1% false-positive budget on validation data, the realised FPR on test data should be close to 1%; any gap is itself "
        "informative. " + T["fpfn"] + " At realistic traffic volumes even a 1% false-positive rate means a large number of daily alerts, so we treat FPR as a first-class "
        "result rather than a footnote (see Section 7.7)."),
      H2("7.5 Detection Latency"),
      P("We time scoring only (feature extraction from raw packets is excluded) and express the result in milliseconds per 1,000 flows (Table IX, Fig. 7). " + T["lat"]),
      P("<b>TABLE IX.</b> Scoring cost.", tcap),
      table([["Model", "ms per 1,000 flows"]] + [[m, V("0.0", "%.1f" % zs.loc[m].latency_ms_per_1k)] for m in MODELS], [2.4 * inch, 1.6 * inch]),
      Spacer(1, 6),
      result_fig("fig_latency.png", "Fig. 7. Scoring latency per model.", 4.2 * inch, 2.0),
      H2("7.6 Feature Relevance"),
      P("To see what the supervised models rely on, we trained a Random Forest on the known-attack scenario and ranked features by Gini importance (Fig. 8). " + T["feat"]),
      result_fig("fig_feature_importance.png", "Fig. 8. Fifteen most important features (Random Forest, known-attack scenario).", 4.4 * inch, 2.6),
      H2("7.7 Operational Implications"),
      P("A false-positive rate looks small until it is multiplied by traffic volume. Table X converts each model's measured mean zero-day FPR into the number of false "
        "alerts per day for three illustrative volumes of mostly benign flows; the volumes are round numbers chosen for illustration, not measurements."),
      P("<b>TABLE X.</b> False alerts per day at illustrative traffic volumes (flows/day \u00d7 mean FPR).", tcap),
      table([["Model", "100,000 flows/day", "1,000,000 flows/day", "10,000,000 flows/day"]] +
            [[m] + [V("n", f"{zs.loc[m].fpr * v:,.0f}") for v in (1e5, 1e6, 1e7)] for m in MODELS], [1.8 * inch, 1.4 * inch, 1.5 * inch, 1.5 * inch]),
      Spacer(1, 6),
      P("Even at one false alert per hundred benign flows, a busy link can generate more alerts than an analyst can read. In practice, a deployment would add alert "
        "aggregation, allow-lists for known-safe hosts and a second-stage check before escalation; the numbers above are therefore an upper bound on analyst workload "
        "rather than a prediction of it."),
      H2("7.8 Discussion"),
      P("The questions below are the ones this study set out to answer."),
      B("<b>Does the model detect attacks absent from training?</b> " + T["q1"]),
      B("<b>Which model gives the best balance?</b> " + T["q2"]),
      B("<b>How does class imbalance affect the results?</b> " + T["q3"]),
      B("<b>Which features matter most?</b> " + T["feat"]),
      B("<b>Does accuracy cost more computation?</b> " + T["q5"]),
      B("<b>Where does the model fail, and why?</b> " + T["q6"]),
      B("<b>How well might it generalise?</b> All results come from one dataset captured in one laboratory network. Performance on a different network, with different "
        "applications and different attack tooling, may differ, and only repeating the study on independent data (for example UNSW-NB15 [[unsw]]) can show by how much.")]

# ---------------------------------------------------------------- 8-10
S += [H1("8. LIMITATIONS"),
      B("<b>Dataset.</b> CIC-IDS2017 is a laboratory capture. It cannot represent the full diversity of real-world zero-day attacks, and it has known labelling issues [[engelen]]."),
      B("<b>Experimental approximation.</b> Holding out an attack family approximates a zero-day scenario. It does not prove that the model would detect any specific "
        "real-world zero-day, and some families share tooling or behaviour with others, so \u201cunseen\u201d is a matter of degree."),
      B("<b>Concept drift.</b> Traffic and attacks change over time and can lower performance after deployment [[gama]]; the results here come from a fixed historical snapshot."),
      B("<b>Adversarial evasion.</b> An adaptive attacker can shape traffic to evade a learned detector; adversarial examples are a known weakness of machine-learning models [[goodfellow]]."),
      B("<b>Operational burden.</b> A 1% false-positive rate can still overwhelm security staff on high-volume links (Section 7.7)."),
      B("<b>Threshold granularity.</b> Tree ensembles produce coarse scores, so reaching exactly 1% FPR is not always possible; the realised FPR is reported for this reason."),
      B("<b>Subsampling.</b> Class sizes were capped for tractability, which may change results slightly compared with training on the full dataset."),
      B("<b>Scope of latency.</b> Latency covers model scoring only and excludes extracting flow features from raw packets; interpretability and deployment cost were not studied in depth."),
      H1("9. FUTURE SCOPE"),
      B("<b>Online and continual learning</b> so the detector adapts as traffic and attacks drift [[gama]]."),
      B("<b>Explainable AI.</b> Feature-attribution methods such as SHAP [[lundberg]] could give analysts a reason for each alert."),
      B("<b>Adversarial robustness.</b> Training against, and evaluating under, evasion attempts [[goodfellow]]."),
      B("<b>Federated learning</b> to let several organisations learn together without sharing raw traffic [[mcmahan]]."),
      B("<b>Graph neural networks</b> [[kipf]] that model relationships between hosts, flows and events rather than isolated flows."),
      B("<b>Real-time deployment</b> on streaming telemetry, and a hybrid of signature and anomaly detection in one pipeline."),
      B("<b>More data.</b> Repeating the study on UNSW-NB15 [[unsw]] and on newer, independently collected traffic."),
      H1("10. CONCLUSION"),
      P("This paper examined whether common machine-learning detectors can recognise attack families that were hidden from them during training, using a "
        "leave-one-attack-family-out protocol on CIC-IDS2017 and a shared 1% false-positive budget. " + T["concl"] +
        " The study shows that evaluation design changes the picture considerably, and that accuracy on a random split should not be read as protection against new "
        "attacks. Because the held-out family is only a stand-in for a real zero-day, the findings should be seen as evidence about generalisation, not a guarantee. "
        "Testing on additional datasets and under adversarial pressure is the next step.")]

# ---------------------------------------------------------------- Refs + appendices
S += [H1("REFERENCES")]
refs_idx = len(S)
S += [H1("APPENDIX A. ABBREVIATIONS"),
      table([["Term", "Meaning"],
             ["IDS / NIDS", "Intrusion detection system / network IDS"], ["ML", "Machine learning"],
             ["FPR / FNR", "False-positive rate / false-negative rate"], ["TPR", "True-positive rate (recall, detection rate)"],
             ["ROC-AUC", "Area under the receiver operating characteristic curve"], ["PR-AUC", "Area under the precision\u2013recall curve"],
             ["DoS / DDoS", "Denial of service / distributed denial of service"], ["XGBoost", "Extreme gradient boosting"],
             ["CICFlowMeter", "Tool that converts packet captures into flow features"]], [1.6 * inch, 4.6 * inch]),
      H1("APPENDIX B. REPRODUCTION STEPS"),
      B("Download the CIC-IDS2017 \u201cMachineLearningCVE\u201d CSV files from the Canadian Institute for Cybersecurity and place them in one folder."),
      B("Install the dependencies: <font name='Courier' size='9'>pip install pandas numpy scikit-learn xgboost matplotlib reportlab pillow</font>."),
      B("Run the experiments: <font name='Courier' size='9'>python run_experiments.py --data MachineLearningCVE --out results</font>."),
      B("Build the paper from the measured results: <font name='Courier' size='9'>python build_paper.py --results results --emails \"a@x.edu,b@x.edu,c@x.edu\"</font>.")]

# numbered references appended in order of first citation
unused = [k for k in REFS if k not in order]
for k in unused: order.append(k)  # keep list complete; unused ones are still listed
ref_flow = [Paragraph("[%d]&nbsp;&nbsp;%s" % (i + 1, REFS[k]), ref) for i, k in enumerate(order)]
S[refs_idx:refs_idx] = ref_flow

doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="AI-Based Zero-Day Attack Detection Using Machine Learning", author="Sumit Maurya, Sumit Vishwakarma, Sunil Gupta")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("Built", args.out, "(%s mode); %d references; unused refs: %s" % ("LIVE" if LIVE else "DRAFT", len(order), unused))
