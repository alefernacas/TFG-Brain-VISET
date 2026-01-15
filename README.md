# Brain-VISET Evaluation Framework for PET Epilepsy Imaging

This repository contains the Python implementation and evaluation framework for **Brain-VISET** (Voxel-based Iterative Simulation for Emission Tomography). This project was developed as a Final Degree Project (TFG) to generate highly realistic [18F]-PET brain databases from clinical refractory epilepsy patients using Monte Carlo simulations.



## 📋 Project Overview

The main objective of this framework is to bridge the gap between theoretical Monte Carlo simulations and clinical reality. By using an iterative feedback loop, the software refines the simulation input until the output is indistinguishable from the patient's real PET scan.

The pipeline consists of two primary stages:
1.  **Anatomical & Activity Pre-processing**: Creating the simulation environment.
2.  **Iterative Optimization**: Matching the simulation to clinical ground truth.

## 🛠 Prerequisites

To use this framework, you need to have the following tools installed:
* **Python 3.8+**
* **SimPET / GATE**: Monte Carlo simulation engine for medical physics.
* **STIR**: Software for Tomographic Image Reconstruction.
* **TotalSegmentator**: AI-based tool for full-body anatomical segmentation.
* **ANTs (Advanced Normalization Tools)**: For high-precision image registration.

## 🚀 Core Scripts

### 1. `Map_Generator.py`
This script prepares the baseline data for each patient.
* **Anatomical Prior**: Uses `TotalSegmentator` to isolate the patient's body and brain.
* **Tissue Segmentation**: Integrates Grey Matter (GM), White Matter (WM), and CSF maps.
* **Data Integrity**: Cleans NaN values and exports maps in **uint8** format, ensuring 100% compatibility with the SimSET/GATE simulation engine.

### 2. `Brain_Viset_Launcher.py`
The execution engine that manages the iterative simulation process.
* **Feedback Loop**: Implements the Brain-VISET algorithm to refine activity maps through multiple iterations.
* **Correction Factor**: Calculates $\text{Factor} = \frac{\text{Real PET}}{\text{Simulated PET}}$ at a voxel-wise level.
* **Spatial Convergence**: Uses ANTs to ensure that the simulated volume is perfectly aligned with the clinical reference before updating the map.



## 📁 Project Structure

```text
├── Data/                   # Patient cohort NIfTI files (MRI, PET, Atlas)
├── Scripts/
│   ├── Map_Generator.py     # Data preparation and labeling
│   └── Brain_Viset_Launcher.py # Iterative simulation & registration loop
├── Results/                # Reconstructed volumes and convergence metrics
└── README.md
