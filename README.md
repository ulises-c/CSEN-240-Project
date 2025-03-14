# CSEN-240 Project

A machine learning project for Santa Clara University’s CSEN/COEN 240 course focused on developing a high-accuracy classifier for Knee Osteoarthritis.

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/) -->

---

## Table of Contents

- [CSEN-240 Project](#csen-240-project)
  - [Table of Contents](#table-of-contents)
  - [How to Run](#how-to-run)
  - [Project Objectives](#project-objectives)
  - [Project Constraints](#project-constraints)
    - [Allowed Changes](#allowed-changes)
    - [Not Allowed Changes](#not-allowed-changes)
  - [Notes \& Code Improvements](#notes--code-improvements)
    - [System Notes](#system-notes)
    - [Changes Made to Base Code](#changes-made-to-base-code)
  - [Monitoring Performance](#monitoring-performance)
  - [Models Evaluated](#models-evaluated)

---

## How to Run

This project uses a Makefile along with a dedicated helper_tools/requirements.txt to ensure a consistent Python environment across different machines.

1. **Install Prerequisites:**

   - Ensure you have [pyenv](https://github.com/pyenv/pyenv) installed.
     - macOS: Installation via Homebrew -> https://formulae.brew.sh/formula/pyenv
     - Linux: Installation instructions on GitHub -> https://github.com/pyenv/pyenv#installation

2. **Clone the Repository:**

   ```
   git clone https://github.com/ulises-c/CSEN-240-Project.git
   ```

3. **Run the Project:**

   - Execute the following command:

   ```
   make run
   ```


   - After running, check the `out/` directory for generated data:
     - Logs
     - Plots
     - Trained Models

---

## Project Objectives

- **Primary Goal:** Develop a machine learning model with high accuracy for classifying Knee Osteoarthritis.
- **Focus Areas:**
  - Data augmentation techniques for the training set.
  - Optimized hyper-parameter tuning.
  - Experimentation with different model architectures.

---

## Project Constraints

The project builds upon a provided base script. Modifications are allowed as follows:

### Allowed Changes

- Adjustments to the ML model architecture.
- Tuning of hyper-parameters.
- Applying image augmentations (training set only), such as:
  - Rotation
  - Noise addition or denoising
  - Diffusion effects
  - Etc.

### Not Allowed Changes

- Any HSV adjustments.
- Crop operations.
- Scale-invariant transformations, including:
  - Color space manipulations (e.g., RGB to greyscale)
  - Resolution adjustments (upscaling or downscaling)
  - Changes to color content or image size

---

## Notes & Code Improvements

### System Notes

- With mixed-precision enabled:
  - RTX 3070: Best stability with a batch_size of 64.
  - M4 Mac Mini: Also tested with batch_size 64.

### Changes Made to Base Code

- **Code Refactoring:**
  - Organized and cleaned up import statements.
  - Converted Jupyter notebook cells to a standard Python script.
- **Helper Tools:**
  - Created scripts for cross-platform compatibility (macOS, Linux).
- **Visualization:**
  - Separated plotting functionality into a dedicated plot_util.py.
- **Logging:**
  - Implemented detailed logging for data, parameters, and training processes to ensure reproducibility.
- **Configuration:**
  - Externalized hyper-parameters and other configuration settings into a JSON file.
- **Learning Rate:**
  - Replaced fixed learning rate with an exponential decay schedule.
- **Data Augmentation:**
  - Enhanced training data using techniques like rotation, vertical/horizontal flip, zoom, shifts, and shearing.
- **Mixed-Precision:**
  - Enabled mixed-precision mode, reducing training time and memory usage.

---

## Monitoring Performance

Monitor GPU usage during training using platform-specific tools:

- Ubuntu: [nvtop](https://github.com/Syllo/nvtop)
- macOS: [asitop](https://github.com/tlkh/asitop)

---

## Models Evaluated

- [x] EfficientNetV2 (Small, Medium, Large)
- [x] Xception
- [x] ResNet50
- [x] ResNet152
- [x] Vision Transformer (ViT)
  - ViT Reference -> https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-b8-classification/1
- [ ] InceptionV3
- [ ] MobileNetV2
- [ ] DenseNet121
- [ ] VGG16
