# Multi-domain Feature Learning for Efficient Multi-scale Underwater Debris Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Official implementation of **YOLO11-MDA**, a multi-domain visual feature learning framework for robust and efficient underwater debris detection.

> This repository is directly related to the manuscript submitted to **Multimedia Systems**.  
> If you find this work useful, please consider citing the corresponding manuscript.

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)


## Overview

Underwater debris detection is important for marine ecological protection, autonomous underwater cleanup, and intelligent underwater robotic systems. However, underwater images are often affected by light attenuation, low contrast, scattering, color distortion, complex background interference, and large scale variations of debris targets.

To address these challenges, this project proposes **YOLO11-MDA**, a robust and efficient underwater debris detection framework based on multi-domain feature learning. The framework jointly exploits spatial-domain details and frequency-domain structural information to improve multi-scale visual representation under degraded underwater imaging conditions.

The proposed method consists of three main components:

- **MDFM**: Multi-Domain Feature Module for spatial-frequency feature representation.
- **DySample**: Dynamic upsampling for fine-grained multi-scale feature fusion.
- **AShape-NWD**: Adaptive shape-aware normalized distance loss for robust localization of small and irregular debris.

---

## Highlights

- Multi-domain visual feature learning for degraded underwater perception.
- Wavelet-based frequency-domain decomposition for structural feature modeling.
- Directional spatial convolution for local detail enhancement.
- Dynamic feature resampling for detail-preserving multi-scale fusion.
- Shape-aware localization loss for small and irregular underwater debris.
- Lightweight and efficient design suitable for real-time underwater perception.

