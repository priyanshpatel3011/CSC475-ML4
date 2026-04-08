# group-project-csc475

# Comparative Analysis of Beat Tracking and Tempo Estimation Algorithms

**Design Specification**

**Authors:** Priyanshkumar Ghanshyambhai Patel, Dharnesh Somasundaram, Yusang Park  
**University:** University of Victoria  
**Date:** February 10, 2026 

---

## Abstract

This design specification outlines a project for comparative analysis of beat tracking and tempo estimation algorithms. We will implement and evaluate three approaches: autocorrelation-based method, dynamic Bayesian networks, and 1D state-space models using GTZAN Tempo-Beat and GiantSteps datasets. The 12-week timeline includes implementation, evaluation, and comprehensive analysis. Expected outcomes include performance comparison across genres, insights into algorithm strengths/limitations, and recommendations for practical applications.

---

## 1. Introduction

Beat tracking and tempo estimation are fundamental MIR tasks enabling applications in music production, recommendation, and education [8]. Despite decades of research, these tasks remain challenging due to musical diversity and rhythmic complexity variations.

### Research Questions

1. How do traditional signal processing approaches compare to modern deep learning methods across genres?
2. What are the computational trade-offs between accuracy and speed?
3. Which musical characteristics most impact performance?
4. Can ensemble methods improve overall performance?

### Scope

We focus on offline global tempo estimation and beat tracking, implementing at least three distinct algorithms for systematic comparison across standardized datasets.

---

## 2. Related Work

Early tempo estimation relied on onset detection and autocorrelation [13]. Percival and Tzanetakis [12] proposed streamlined autocorrelation with pulse cross-correlation, achieving competitive results with reduced complexity. Recent work explores deep learning [14] with efficient implementations in madmom [1].

Beat tracking evolved from rule-based systems to probabilistic models. Ellis [5] introduced dynamic programming for beat tracking. Böck and Schedl [3] enhanced performance using context-aware neural networks. Recent work integrates beat and downbeat tracking [2] and explores transformer architectures [6].

### Key Datasets

- **GTZAN Tempo-Beat** [10]: 1000 30-second excerpts across 10 genres
- **GiantSteps** [9]: 664 EDM tracks with expert-verified tempo annotations
- **Ballroom** [8]: Classical benchmark dataset

### Evaluation Metrics

- **Tempo:** ACC1/ACC2
- **Beat Tracking:** F-measure

### Recent Advances

Recent advances include self-supervised learning [7], adapter tuning [4], transformer architectures [6], and zero-latency systems [11]. Challenges remain in cross-dataset generalization, computational efficiency, and systematic comparison of classical vs. modern approaches.

---

## 3. Methodology

### 3.1 Datasets

**GTZAN Tempo-Beat:** 1000 audio excerpts (30s each) across 10 genres with tempo annotations and beat positions. Valuable for genre-specific evaluation.

**GiantSteps:** 664 EDM tracks with expert-verified tempo annotations, focusing on rhythmically complex patterns and higher tempo ranges (120-180 BPM).

### 3.2 Algorithms

1. **Autocorrelation-Based**
   - Following [12], we implement onset strength envelope computation, autocorrelation, and pulse cross-correlation
   - Serves as classical signal processing baseline

2. **Dynamic Bayesian Network**
   - Using madmom [1], we utilize RNN-based beat activation with DBN and particle filtering
   - Represents modern probabilistic approaches

3. **1D State-Space Model**
   - Implements dimensionality reduction with semi-Markov model and jump-back reward strategy
   - Optimized for computational efficiency

### 3.3 Evaluation

**Tempo Metrics:**
- ACC1 (4% tolerance)
- ACC2 (allowing octave errors)
- P-Score
- Mean Absolute Error

**Beat Metrics:**
- F-measure (70ms tolerance)
- Cemgil Accuracy
- Information Gain

**Analysis:**
- Per-genre comparison
- Tempo range analysis
- Computational benchmarks
- Error categorization
- Statistical significance testing

### 3.4 Implementation

**Data Preprocessing:**
- 44.1kHz mono conversion
- 70/15/15 train/validation/test split
- Feature extraction (spectrograms, onset envelopes, beat activations)

**Software Stack:**
- Librosa 0.10.x
- madmom 0.16.x
- Custom state-space implementation
- SciPy/NumPy for analysis
- Git for version control
- LaTeX for documentation

---

## 4. Team Organization

### Roles & Responsibilities

**Priyanshkumar Ghanshyambhai Patel - Project Lead & Algorithm Implementation**
- PI1 (basic): Schedule and chair weekly team meetings; maintain a shared task tracker updated after every meeting
- PI2 (advanced): Implement autocorrelation baseline and 1D state-space model achieving ≥70% ACC1 on the GTZAN test set by Week 6
- PI3 (expected): Complete a signal processing literature review covering ≥10 papers, summarized in a dedicated section of the final report

**Dharnesh Somasundaram - Data & Evaluation**
- PI1 (expected): Acquire and preprocess both GTZAN and GiantSteps datasets into a clean pipeline with 70/15/15 splits completed by Week 4
- PI2 (advanced): Develop a unified evaluation framework that correctly computes ACC1, ACC2, F-measure, and Cemgil Accuracy across all three algorithms
- PI3 (basic): Configure and verify the madmom DBN implementation with correct input/output format by Week 7
- PI4 (expected): Perform statistical significance testing (Wilcoxon signed-rank test) on all cross-algorithm performance comparisons

**Yusang Park - Analysis & Documentation**
- PI1 (basic): Produce per-genre and per-algorithm visualization plots (bar charts, scatter plots) for all evaluation metrics by Week 10
- PI2 (expected): Investigate and document ≥5 failure cases per algorithm with written root cause analysis
- PI3 (basic): Maintain up-to-date README and inline code documentation throughout the entire project duration
- PI4 (advanced): Author the complete final technical report in LaTeX, integrating all results, visualizations, analysis, and practical recommendations

### Collaboration

- Weekly Monday meetings at 3 PM
- GitHub repository for version control
- Shared Overleaf document
- Slack communication
- Mandatory code reviews

---

## 5. Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1-2 | Setup | Literature review, datasets, environment, design spec |
| 3-4 | Preprocessing | Feature pipeline, EDA, baseline architecture |
| 5-6 | Baseline | Autocorrelation implementation, initial evaluation |
| 7-8 | Advanced | DBN and state-space models, cross-validation |
| 9-10 | Evaluation | Test set analysis, statistics, error analysis |
| 11 | Documentation | Final report, visualizations, presentation |
| 12 | Finalization | Submission, code documentation |

### Key Milestones

- **Week 2:** Design specification complete
- **Week 4:** Preprocessing complete
- **Week 6:** Baseline functional
- **Week 8:** All algorithms implemented
- **Week 10:** Evaluation complete
- **Week 12:** Final submission

---

## 6. Expected Outcomes

### Technical Deliverables

- Three algorithm implementations with unified evaluation framework and preprocessing pipeline
- Comprehensive performance comparison with genre-specific analysis and computational benchmarks
- Well-documented code, technical report, and Jupyter notebooks

### Research Contributions

- Systematic comparison using identical evaluation framework
- Insights into genre-specific performance
- Analysis of accuracy vs. efficiency trade-offs
- Identification of failure modes
- Algorithm selection recommendations

### Learning Objectives

- Hands-on MIR experience
- Python scientific computing proficiency
- Reproducible research practices
- Collaborative development
- Technical communication skills

---

## 7. Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| **Dataset Access** | If GTZAN unavailable, use Ballroom/Hainsworth datasets |
| **Implementation Complexity** | Allocate buffer time in weeks 7-8, simplify state-space model if needed |
| **Computational Resources** | Use university clusters, optimize code efficiency |
| **Team Coordination** | Flexible meeting times, clear communication protocols |

---

## 8. Project Structure

csc475/
├── data/
│   ├── raw/            # Original downloaded datasets
│   ├── processed/      # Resampled 44.1kHz mono WAV files
│   └── features/       # Precomputed .npy feature files
├── src/
│   ├── algorithms/
│   │   ├── autocorrelation.py      # Algorithm 1: Classical baseline
│   │   ├── dbn_tracker.py          # Algorithm 2: Dynamic Bayesian Network
│   │   └── state_space.py          # Algorithm 3: 1D State-Space Model
│   ├── evaluation/
│   │   ├── metrics.py              # ACC1, ACC2, F-measure, Cemgil, etc.
│   │   ├── evaluator.py            # Run evaluation across all algorithms
│   │   └── statistical_tests.py   # Wilcoxon signed-rank tests
│   ├── visualization/
│   │   └── plots.py                # All plots and figures
│   └── utils/
│       ├── audio.py                # Audio loading and preprocessing
│       ├── dataset.py              # Dataset loading and splitting
│       └── config.py               # Global config constants
├── scripts/
│   ├── preprocess.py               # Run full data preprocessing pipeline
│   ├── extract_features.py         # Run full feature extraction pipeline
│   ├── run_evaluation.py           # Run all algorithms on test set
│   └── run_all.py                  # Master script: run everything end-to-end
├── tests/
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── notebooks/
│   └── analysis.ipynb              # EDA and results analysis
├── results/
│   ├── figures/                    # Output plots
│   └── metrics/                    # JSON results files
├── requirements.txt
└── README.md

---
## 9. Setup
bash
# Create conda environment
conda create -n csc475 python=3.10
conda activate csc475
# Install dependencies
pip install -r requirements.txt
# Verify installation
python -c "import librosa, madmom, mir_eval; print('All packages OK')"

---
## 10. Usage
### 1. Preprocess Data
bash
python scripts/preprocess.py --input data/raw/gtzan --output data/processed/gtzan
python scripts/preprocess.py --input data/raw/giantsteps --output data/processed/giantsteps

### 2. Extract Features
bash
python scripts/extract_features.py --input data/processed --output data/features

### 3. Run Full Evaluation
bash
python scripts/run_evaluation.py --dataset gtzan --split test
python scripts/run_evaluation.py --dataset giantsteps --split test

### 4. Run Everything (end-to-end)
bash
python scripts/run_all.py

---
## 11. Algorithms
| Algorithm | Method | Expected ACC1 |
|---|---|---|
| Autocorrelation | Classical signal processing baseline | ≥70% |
| DBN (madmom) | RNN + Dynamic Bayesian Network | ≥80% |
| State-Space | Semi-Markov + Viterbi decoding | ≥75% |
---
## 12. Running Tests & Conclusion

```bash
pytest tests/ -v
```

This specification outlines a systematic comparison of beat tracking and tempo estimation algorithms spanning classical to modern approaches. The structured timeline with clear milestones and defined team roles ensures comprehensive coverage while maintaining flexibility. Expected outcomes include meaningful insights into algorithm characteristics and practical recommendations for MIR applications.



This specification outlines systematic comparison of beat tracking and tempo estimation algorithms spanning classical to modern approaches. The structured timeline with clear milestones and defined team roles ensures comprehensive coverage while maintaining flexibility. Expected outcomes include meaningful insights into algorithm characteristics and practical recommendations for MIR applications.

---

## References

[1] Sebastian Böck, Filip Korzeniowski, Jan Schlüter, Florian Krebs, and Gerhard Widmer. madmom: A new Python audio and music signal processing library. In *Proc. of the 24th ACM Int. Conf. on Multimedia*, pages 1174–1178, 2016.

[2] Sebastian Böck, Florian Krebs, and Gerhard Widmer. Joint beat and downbeat tracking with recurrent neural networks. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 255–261, 2016.

[3] Sebastian Böck and Markus Schedl. Enhanced beat tracking with context-aware neural networks. In *Proc. Int. Conf. Digital Audio Effects (DAFx)*, pages 135–139, 2011.

[4] Jiajun Deng, Yaolong Ju, Jing Yang, Simon Lui, and Xunying Liu. Efficient adapter tuning for joint singing voice beat and downbeat tracking with self-supervised learning features. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 343–351, 2024.

[5] Daniel PW Ellis. Beat tracking by dynamic programming. Volume 36, pages 51–60, 2007.

[6] Francesco Foscarin, Jan Schlüter, and Gerhard Widmer. Beat this! accurate beat tracking without DBN postprocessing. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 962–969, 2024.

[7] Antonin Gagnère, Slim Essid, and Geoffroy Peeters. A contrastive self-supervised learning scheme for beat tracking amenable to few-shot learning. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 198–206, 2024.

[8] Fabien Gouyon, Anssi Klapuri, Simon Dixon, Miguel Alonso, George Tzanetakis, Christian Uhle, and Pedro Cano. An experimental comparison of audio tempo induction algorithms. *IEEE Transactions on Audio, Speech, and Language Processing*, 14(5):1832–1844, 2006.

[9] Peter Knees, Ángel Faraldo, Herrera Boyer, Richard Vogl, Sebastian Böck, Felix Hörschläger, Mickael Le Goff, and Mohamed Sordo. Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 364–370, 2015.

[10] Ugo Marchand and Geoffroy Peeters. GTZAN-rhythm: Extending the GTZAN test-set with beat, downbeat and swing annotations. In *Proc. of the Int. Society for Music Information Retrieval Conf. (ISMIR)*, pages 627–633, 2015.

[11] Philipp Meier, Ching-Yu Chiu, and Meinard Müller. A real-time beat tracking system with zero latency and enhanced controllability. *Transactions of the International Society for Music Information Retrieval*, 7(1):213–227, 2024.

[12] Graham Percival and George Tzanetakis. Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 22(12):1765–1776, 2014.

[13] Eric D Scheirer. Tempo and beat analysis of acoustic musical signals. *The Journal of the Acoustical Society of America*, 103(1):588–601, 1998.

[14] Hendrik Schreiber and Meinard Müller. Musical tempo and key estimation using convolutional neural networks with directional filters. In *Proc. of the Sound and Music Computing Conf.*, pages 47–54, 2018.

---


## Contact

- Priyanshkumar Ghanshyambhai Patel: priyanshpatel0211@gmail.com 
- Dharnesh Somasundaram: dharneshsomasundaram2001@gmail.com 
- Yusang Park: usang.park@gmail.com
