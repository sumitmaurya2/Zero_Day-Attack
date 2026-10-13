#!/usr/bin/env python3
"""
Zero-day generalisation experiments on CIC-IDS2017 (MachineLearningCVE CSVs).

Usage:
    python run_experiments.py --data /path/to/MachineLearningCVE --out results

Protocol
--------
1. Known-attack scenario : every attack family appears in training and test.
2. Zero-day scenario     : leave-one-attack-family-out. One family is removed
                           from training completely and appears only in the test set.
3. Operating point       : for EVERY model the alert threshold is the 99th
                           percentile of the score on a held-out *benign validation* set,
                           so all models are compared at a ~1% false-positive budget.
Everything printed or saved by this script is measured, not assumed.
"""
import argparse, glob, json, os, platform, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

SEED = 42
FAMILIES_TO_HOLD_OUT = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]


# ------------------------------------------------------------------ data
def family_of(label: str) -> str:
    l = label.strip().lower()
    if l == "benign": return "BENIGN"
    if l.startswith("web attack"): return "WebAttack"
    if l in ("ftp-patator", "ssh-patator"): return "BruteForce"
    if l.startswith("dos") or l == "heartbleed": return "DoS"
    if l == "ddos": return "DDoS"
    if l == "portscan": return "PortScan"
    if l == "bot": return "Bot"
    return "Infiltration"


def load_data(folder, max_benign, max_per_family):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["family"] = df["Label"].astype(str).map(family_of)
    df = df.drop(columns=["Label"])
    for c in ("Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port"):
        if c in df.columns: df = df.drop(columns=[c])
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()
    df = df.loc[:, (df.nunique() > 1) | (df.columns == "family")]
    parts = []
    for fam, g in df.groupby("family"):
        cap = max_benign if fam == "BENIGN" else max_per_family
        parts.append(g.sample(min(len(g), cap), random_state=SEED))
    df = pd.concat(parts, ignore_index=True)
    # Infiltration has only a few dozen flows: too few for held-out evaluation.
    df = df[df["family"] != "Infiltration"].reset_index(drop=True)
    return df


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


# ------------------------------------------------------------------ models
class Supervised:
    def __init__(self, est): self.est = est
    def fit(self, X, y, Xb, Xv): self.est.fit(X, y); return self
    def score(self, X): return self.est.predict_proba(X)[:, 1]


class IsoForest:
    def fit(self, X, y, Xb, Xv):
        self.m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xb); return self
    def score(self, X): return -self.m.score_samples(X)


class AutoEncoder:
    def fit(self, X, y, Xb, Xv):
        Xb = Xb[:100000]
        self.m = MLPRegressor(hidden_layer_sizes=(32, 16, 32), activation="relu", max_iter=40,
                              early_stopping=True, random_state=SEED).fit(Xb, Xb); return self
    def score(self, X): return ((self.m.predict(X) - X) ** 2).mean(axis=1)


class Hybrid:
    """XGBoost probability fused (max) with a benign-only autoencoder anomaly score."""
    def fit(self, X, y, Xb, Xv):
        self.sup = Supervised(XGBClassifier(**XGB_PARAMS)).fit(X, y, Xb, Xv)
        self.ae = AutoEncoder().fit(X, y, Xb, Xv)
        self.ref = np.quantile(self.ae.score(Xv), 0.99); return self
    def score(self, X):
        z = self.ae.score(X) / self.ref
        return np.maximum(self.sup.score(X), z / (1 + z))


XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1)


def make_models():
    return {
        "Logistic Regression": Supervised(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Decision Tree": Supervised(DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=SEED)),
        "Random Forest": Supervised(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                           n_jobs=-1, random_state=SEED)),
        "XGBoost": Supervised(XGBClassifier(**XGB_PARAMS)),
        "Isolation Forest": IsoForest(),
        "Autoencoder": AutoEncoder(),
        "Proposed Hybrid": Hybrid(),
    }


# ------------------------------------------------------------------ evaluation
def evaluate(model, Xv_b, Xte, yte):
    thr = np.quantile(model.score(Xv_b), 0.99)
    t0 = time.perf_counter(); s = model.score(Xte); dt = time.perf_counter() - t0
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    tn = int(((pred == 0) & (yte == 0)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
    return dict(precision=precision_score(yte, pred, zero_division=0),
                recall=recall_score(yte, pred), f1=f1_score(yte, pred, zero_division=0),
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1),
                roc_auc=roc_auc_score(yte, s), pr_auc=average_precision_score(yte, s),
                latency_ms_per_1k=1000 * dt / len(yte) * 1000, TP=tp, FP=fp, TN=tn, FN=fn), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--max-benign", type=int, default=200000)
    ap.add_argument("--max-per-family", type=int, default=30000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = load_data(a.data, a.max_benign, a.max_per_family)
    feats = [c for c in df.columns if c != "family"]
    X_all = signed_log(df[feats].to_numpy(dtype=float)); fam = df["family"].to_numpy()
    print("Class counts:\n", df["family"].value_counts())

    # benign: 60/20/20 train/val/test ; attacks: 70/30 train/test per family
    ben = np.where(fam == "BENIGN")[0]
    b_tr, b_rest = train_test_split(ben, test_size=0.4, random_state=SEED)
    b_va, b_te = train_test_split(b_rest, test_size=0.5, random_state=SEED)
    att_tr, att_te = {}, {}
    for f in sorted(set(fam) - {"BENIGN"}):
        idx = np.where(fam == f)[0]
        att_tr[f], att_te[f] = train_test_split(idx, test_size=0.3, random_state=SEED)

    comp = [dict(cls="BENIGN", total=len(ben), train=len(b_tr), val=len(b_va), test=len(b_te))]
    for f in sorted(att_tr):
        comp.append(dict(cls=f, total=len(att_tr[f]) + len(att_te[f]), train=len(att_tr[f]), val=0, test=len(att_te[f])))
    pd.DataFrame(comp).to_csv(f"{a.out}/dataset_composition.csv", index=False)
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        ram = float("nan")
    import sklearn, xgboost
    json.dump(dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                   numpy=np.__version__, pandas=pd.__version__, os=platform.platform(),
                   cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                   ram_gb=round(ram, 1), gpu="not used", n_features=len(feats), seed=SEED),
              open(f"{a.out}/environment.json", "w"), indent=2)

    def run(train_fams, test_idx_attack, tag, holdout=None):
        tr = np.concatenate([b_tr] + [att_tr[f] for f in train_fams])
        sc = StandardScaler().fit(X_all[tr])
        Xtr, ytr = sc.transform(X_all[tr]), (fam[tr] != "BENIGN").astype(int)
        Xb, Xv = sc.transform(X_all[b_tr]), sc.transform(X_all[b_va])
        te = np.concatenate([b_te, test_idx_attack]); Xte = sc.transform(X_all[te])
        yte = (fam[te] != "BENIGN").astype(int)
        rows, scores = [], {}
        for name, m in make_models().items():
            print(f"[{tag}] training {name} ...", flush=True)
            m.fit(Xtr, ytr, Xb, Xv)
            r, s = evaluate(m, Xv, Xte, yte)
            rows.append(dict(scenario=tag, heldout=holdout or "-", model=name, **r)); scores[name] = (yte, s)
        return rows, scores

    all_fams = sorted(att_tr)
    known_rows, _ = run(all_fams, np.concatenate([att_te[f] for f in all_fams]), "known")
    pd.DataFrame(known_rows).to_csv(f"{a.out}/known_attack_results.csv", index=False)

    zd_rows, curves = [], {}
    for f in [x for x in FAMILIES_TO_HOLD_OUT if x in att_tr]:
        # held-out family: ALL its flows are unseen at training time
        rows, sc_ = run([x for x in all_fams if x != f], np.concatenate([att_tr[f], att_te[f]]),
                        "zero_day", holdout=f)
        zd_rows += rows; curves[f] = sc_
    zd = pd.DataFrame(zd_rows); zd.to_csv(f"{a.out}/zero_day_results.csv", index=False)

    summ = (zd.groupby("model")[["precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc",
                                 "latency_ms_per_1k"]].mean().round(4))
    summ.to_csv(f"{a.out}/zero_day_summary_mean_over_families.csv")
    print("\nZero-day summary (mean over held-out families):\n", summ)

    # ---------------- figures
    models = list(summ.index)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (f, d) in zip(axs.ravel(), curves.items()):
        for m in models:
            y, s = d[m]; fpr, tpr, _ = roc_curve(y, s); ax.plot(fpr, tpr, label=m, lw=1.2)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_title(f"Held-out: {f}", fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{a.out}/fig_roc_zero_day.png", dpi=200); plt.close()

    piv = zd.pivot(index="heldout", columns="model", values="recall")[models]
    piv.plot(kind="bar", figsize=(11, 5)); plt.ylabel("Detection rate (recall) at ~1% FPR")
    plt.title("Zero-day detection rate by held-out attack family"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_model_comparison.png", dpi=200); plt.close()

    summ["latency_ms_per_1k"].plot(kind="barh", figsize=(7, 4)); plt.xlabel("ms per 1,000 flows (scoring only)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_latency.png", dpi=200); plt.close()

    best = summ["f1"].idxmax(); c = zd[zd.model == best][["TP", "FP", "TN", "FN"]].sum()
    cm = np.array([[c.TN, c.FP], [c.FN, c.TP]])
    plt.figure(figsize=(4.2, 3.8)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    plt.xticks([0, 1], ["Benign", "Attack"]); plt.yticks([0, 1], ["Benign", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best} (summed over folds)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_confusion_matrix.png", dpi=200); plt.close()

    # feature importance (Random Forest, known-attack scenario) -> Discussion Q4
    tr = np.concatenate([b_tr] + [att_tr[f] for f in all_fams]); sc = StandardScaler().fit(X_all[tr])
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")
    rf.fit(sc.transform(X_all[tr]), (fam[tr] != "BENIGN").astype(int))
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values().tail(15)
    imp.sort_values(ascending=False).to_csv(f"{a.out}/feature_importance.csv", header=["importance"])
    imp.plot(kind="barh", figsize=(7, 5)); plt.xlabel("Gini importance"); plt.tight_layout()
    plt.savefig(f"{a.out}/fig_feature_importance.png", dpi=200); plt.close()
    print(f"\nDone. Everything is in ./{a.out}/")


if __name__ == "__main__":
    main()
