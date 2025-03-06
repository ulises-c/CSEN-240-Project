# CSEN-240-Project
Project for Santa Clara University (SCU) CSEN/COEN 240

## Project Objectives
1. Create a model with high accuracy for classifying Knee Osteoarthritis

## Project Constrains
Base script was given and the following changes are either allowed or not allowed.

<br>

Allowed
1. ML model
2. Hyper-parameters
3. Image augmentations (only to training set)
   1. Rotation
   2. Noise / denoise
   3. Diffusion
   4. etc.
   
<br>

Not allowed
1. Scale invariant transformations
   1. No color space manipulation (e.g. RGB -> greyscale)
   2. Resolution changes (upscale or downscale)
   3. Cropping
   4. Color content
   5. Size
   6. etc.

## Notes
1. With `mixed-precision` on an RTX 3070 and M4 Mac Mini `batch_size` is most stable at 64 (from current testing)

### Changes made to base code
Not in any particular order
1. Cleaned up imports
2. Changed from notebook cell style to traditional python script
3. Created helper tools to have this run on macOS (Apple Silicon) and Linux
   1. More on that in `How to run` section
4. Created `plot_util.py` instead of keeping plots in `knee-osteo.py`
5. Logging data and parameters to have traceability so that models can easily be recreated
6. Made hyper-parameters and other config parameter read from a JSON file
7. Changed learning rate to use exponential decay instead a fixed learning rate
8. Augmented training data (rotation, vertical flip, horizontal flip, zoom, height shift, width shift, shearing, etc.)
9. Enabled `mixed-precision` mode
   1.  This was a big game changer, reduced training time heavily by allowing better use of more recent hardware
   2.  Increased `batch_size` as a result of reduced VRAM usage


### How to run
Use the `Makefile` which utilizes `helper_tools/requirements.txt` to create a consistent python environment across machines
1. Make sure your host machine has `pyenv` installed
   1. macOS - [github](https://github.com/pyenv/pyenv?tab=readme-ov-file#macos) | [brew](https://formulae.brew.sh/formula/pyenv)
   2. Linux - [github](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)
2. Clone the repo
3. In the terminal, with the project directory open, enter the following command
   1. `make run`
   2. `out/` will contain output data from model training, such as the following:
      1. logs
      2. plots
      3. models

## Monitor performance
Monitor GPU usage as you train with one of the following tools depending on your platform
- Ubuntu: nvtop - [github](https://github.com/Syllo/nvtop?tab=readme-ov-file#nvtop)
- macOS: asitop - [github](https://github.com/tlkh/asitop) | [brew](https://formulae.brew.sh/formula/asitop)