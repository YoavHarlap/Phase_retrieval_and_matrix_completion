# Phase Retrieval and Matrix Completion through projection-based algorithms

## Overview
This repository contains the code and numerical experiments from the thesis "Phase Retrieval and Matrix Completion through projection-based algorithms". The project investigates two fundamental problems in signal processing:

1. **Phase Retrieval**: Recovering the global phase vector from Fourier magnitude data.
2. **Matrix Completion**: Reconstructing low-rank matrices from partial data.

The research systematically compares the performance of projection-based algorithms under varying conditions, highlighting trade-offs in reliability, computational efficiency, and scalability.

---

## Features
- Implementation of projection-based algorithms for phase retrieval and matrix completion.
- Evaluation of algorithmic performance under various scenarios:
  - **Random Phase Retrieval**
  - **Crystallographic Phase Retrieval**
  - **Matrix Completion with Low-Rank Constraints**
- Analysis of robustness to noise and effects of missing data.
- Visualization of convergence behavior and performance metrics.

---

## Installation

Clone the repository:
```bash
$ git clone https://github.com/YoavHarlap/Phase_retrieval_and_matrix_completion.git
$ cd Phase_retrieval_and_matrix_completion
```

Install required dependencies:
```bash
$ pip install -r requirements.txt
```

---
## Repository Structure
- **Random_phase_retrieval.py**: Implements Random_phase_retrieval experiments.
- **Crystallographic_phase_retrieval**: Implements Crystallographic_phase_retrieval experiments.
- **matrix_completion.py**: Implements matrix completion experiments.
- **Introduction.py**: Implementation of the images in the introduction that demonstrate the importance of the phase.
- **requirements.txt**: List of dependencies.

---

## Results
In order to perform any of the experiments conducted during the thesis, one only needs to change the appropriate parameters in the code.

---

## Contact 
For any further inquiries or collaboration opportunities, please reach out to:

Yoav Harlap
- Email: yoavharlap@mail.tau.ac.il
- GitHub: YoavHarlap (https://github.com/YoavHarlap)


Thank you for your interest in our project! 🚀
