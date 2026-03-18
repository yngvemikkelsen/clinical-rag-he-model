# clinical-rag-he-model

Health economic model for the paper:

> Mikkelsen Y. **Early economic evaluation of retrieval-layer correction in clinical RAG: a decision-uncertainty framework.** *Value in Health.* 2026 [in press].

---

## Overview

This repository contains the decision model code accompanying the above paper. The model estimates the economic plausibility of corpus-only ZCA whitening as a post-hoc correction for degraded embedding geometry in clinical retrieval-augmented generation (RAG) pipelines.

**Primary comparison:** Whitening (B) vs. no correction (A)  
**Perspective:** Norwegian healthcare system (NoMA reference case)  
**Horizon:** 5 years  
**Discount rate:** 4% (3.0% and 3.5% sensitivity available)

---

## Key parameters

| Parameter | Base case | Source |
|---|---|---|
| Surrogate link α | 1.111 | DiReCT empirical (ClinicalBERT, MIMIC-IV) |
| ΔMRR — heterogeneous corpus | +0.221 | Mikkelsen 2026 JAMIA |
| ΔMRR — homogeneous corpus (harm) | −0.05 | Mikkelsen 2026 JAMIA |
| Conservative α | 0.36 | Tao et al. 2020 (CDSS transported) |
| Cost per ADE (system) | €12,000 | Norwegian DRG + OECD |
| Whitening implementation cost | €800 | Engineering estimate |

---

## Files

```
clinical_rag_he_model_v2.py    Main model (base case, PSA, thresholds, scenarios, figures)
notebooks/
  direct_alpha_experiment.ipynb  DiReCT α estimation experiment (GPU, Colab)
```

---

## Requirements

```
numpy
pandas
matplotlib
scipy
```

Standard Python 3.10+ environment. No GPU required for the HE model.  
The `direct_alpha_experiment.ipynb` notebook requires GPU (tested on A100 via Colab Pro+).

---

## Usage

```bash
pip install numpy pandas matplotlib scipy
python clinical_rag_he_model_v2.py
```

Outputs:
- Printed results table (base case, scenarios, thresholds, PSA, sensitivity)
- `he_model_results_v2.png` — four-panel figure (CE plane, CEAC, tornado, α sensitivity)

---

## DiReCT α experiment

The surrogate link α = 1.111 was estimated from the [MIMIC-IV-Ext-DiReCT dataset](https://physionet.org/content/mimic-iv-ext-direct/1.0.0/) (PhysioNet, credentialed access required). The experiment notebook (`notebooks/direct_alpha_experiment.ipynb`) encodes 343 MIMIC-IV clinical notes with BioBERT, ClinicalBERT, BGE-base, and GTE-base, applies ZCA whitening, and measures ΔMRR@10 and Δ top-1 PDD retrieval accuracy to estimate α per model. ClinicalBERT PDD-level α = 1.111 (95% CI [1.014, 2.541]) is used as the base case.

**Important:** The DiReCT experiment measures diagnosis-label retrieval accuracy, not clinician diagnostic accuracy. Whether retrieval improvement translates to clinical diagnostic performance remains an unvalidated structural assumption — this is the paper's central uncertainty.

---

## Companion repositories

- **clinical-rag-benchmark**: MRR@10 benchmark data across 294 conditions, 11 models, 3 corpora — [github.com/yngvemikkelsen/clinical-rag-benchmark](https://github.com/yngvemikkelsen/clinical-rag-benchmark)
- **clinical-embedding-fix**: Python library implementing corpus-only ZCA whitening — [github.com/yngvemikkelsen/clinical-embedding-fix](https://github.com/yngvemikkelsen/clinical-embedding-fix)

---

## Citation

```bibtex
@article{mikkelsen2026rag_he,
  author  = {Mikkelsen, Yngve},
  title   = {Early economic evaluation of retrieval-layer correction in
             clinical {RAG}: a decision-uncertainty framework},
  journal = {Value in Health},
  year    = {2026},
  note    = {In press}
}
```

---

## Data use

The DiReCT dataset (MIMIC-IV-Ext-DiReCT) is subject to the PhysioNet Credentialed Health Data License. Raw data cannot be shared. Only derived statistics (α estimates, bootstrap CIs) are reported here.

---

## License

Code: MIT  
Data: see PhysioNet DUA for MIMIC-IV-Ext-DiReCT
