# Classical vs Quantum Machine Learning for Cyberattack Detection in Smart Grids

## Overview
This research project compares classical machine learning (ML) and quantum machine 
learning (QML) models for detecting cyberattacks in US smart grid systems. The core 
question: Can QML achieve faster and more accurate malicious activity detection than 
classical ML?

## Research Question
Can Quantum Machine Learning (QML) outperform Classical ML in detecting cyberattacks 
vs. legitimate activity in smart grids in terms of accuracy, precision, and speed?

## Models Compared
**Classical ML:**
- Support Vector Machine (SVM)

**Quantum ML:**
- Quantum Support Vector Machine (QSVM)

## Technologies Used
- Python
- qiskit (quantum simulation)
- Scikit-learn (classical ML)
- Jupyter Notebooks

## Evaluation Metrics
- Accuracy
- Precision
- Training/Testing Speed

## Project Structure
classical-vs-quantum-ml-smart-grid-security/
├── classical_ml      # SVM code
├── quantum_ml        # QSVM and QNN simulations
├── data               # Dataset info (see data/README.md)
├── results            # Evaluation outputs
├── requirements.txt
└── README.md

## Dataset
This project uses the Edge-IIoTset dataset. See data/README.md for download instructions.

## How to Run
pip install -r requirements.txt
python svm.py
python qsvm.py

## Researchers
- **Tracey Osei Kwarteng** - Dataset collection and classical ML simulation
- **Grace Fidele Dushime** - Quantum ML simulation
- **Faculty Mentor:** Dr Mariam Gado

## Status
In progress
