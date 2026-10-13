
# AI-Based Zero-Day Attack Detection Using Machine Learning

### A Hybrid Supervised–Anomaly Framework and a Leave-One-Attack-Family-Out Evaluation Protocol

## 📌 Overview

Zero-day attacks exploit previously unknown vulnerabilities, making them difficult for traditional signature-based Intrusion Detection Systems (IDS) to identify.

This research proposes a machine learning-based framework for detecting previously unseen attack patterns by combining supervised attack classification with unsupervised anomaly detection.

The proposed framework integrates:

- **XGBoost Classifier:** Identifies malicious traffic patterns learned from known attack families.
- **Autoencoder:** Learns the characteristics of benign network traffic and identifies potentially anomalous behaviour.
- **Hybrid Detection:** Combines both models' scores to detect known attacks and investigate unfamiliar traffic patterns.

The framework is designed to evaluate generalisation to unseen attack families while maintaining a consistent false-positive budget.

**Important:** This research paper presents a proposed methodology and experimental evaluation protocol. It does not report measured experimental results or claim proven real-world zero-day detection capability.

---

## 🎯 Research Objectives

- Design a machine learning framework for detecting previously unseen cyberattack patterns.
- Develop a reproducible preprocessing pipeline for network traffic data.
- Compare supervised and unsupervised intrusion detection approaches.
- Evaluate detection performance when an entire attack family is excluded from training.
- Maintain a common false-positive budget across competing models.
- Analyse detection rates, false alarms, ranking quality, and computational cost.

---

## 🧠 Proposed System Architecture

The framework processes network flow records through two complementary detection branches.

```text
             CIC-IDS2017 Dataset
                      |
                      v
           Data Preprocessing
                      |
             Feature Transformation
                      |
             +--------+--------+
             |                 |
             v                 v
       XGBoost Classifier   Autoencoder
       Known Attacks       Benign Traffic
             |                 |
             v                 v
       Attack Score       Anomaly Score
             |                 |
             +--------+--------+
                      |
                      v
                Score Fusion
                      |
                      v
             Threshold Evaluation
                      |
             +--------+--------+
             |                 |
             v                 v
          Alert             Pass
```
