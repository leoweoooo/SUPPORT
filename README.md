# support
Self-supervised denoising method for voltage imaging data, as described and published by [NICALab](https://github.com/NICALab/SUPPORT).

This is a fork of their original repository with quality-of-life improvements. The changes made so far are:

- command-line argument parsing for both `train.py` and `test.py`, along with refactoring for modular code.
- saving training conditions from `train.py` to allow `test.py` to read and apply the right model params.
- removed CUDA-specific code to allow for non-CUDA devices. 
- deleted duplicate code and restructured the repository for easier maintenance.

There are more in plan, including:

- a loading script that converts video files into the right format for the models.
- web-based GUI with simpler dependencies.
