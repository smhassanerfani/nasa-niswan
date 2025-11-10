# Smart-NINT: Spatio-Temporal Machine Learning Approaches for Atmospheric Composition Emulation in NASA GISS ModelE

<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://arxiv.org/pdf/2510.10654" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2408.11032-b31b1b.svg" alt="ArXiv">
</a>
<a href="https://doi.org/10.5281/zenodo.17575784">
    <img src="https://zenodo.org/badge/771101661.svg" alt="DOI"></a>

This repository presents **Smart NINT**, a spatiotemporal machine learning (ML) framework designed to emulate the computationally expensive process of **interactive atmospheric composition** transport within Earth System Models (ESMs), specifically tested using the NASA GISS ESM V3.0 (ModelE).

***

## Installation

```
conda create -n niswan -c conda-forge python=3.12
conda activate niswan
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda -c conda-forge scikit-learn seaborn xesmf
pip install torch torchvision lightning tqdm
```


## Overview

ESMs often rely on prescribed monthly tracer concentration fields (**Non-Interactive Tracers, NINT**) to save computational resources, which sacrifices the ability to capture real-time aerosol-climate feedbacks. Smart NINT addresses this limitation by using a **spatiotemporal ML architecture** to interactively calculate tracer concentrations, such as **Black Carbon from Biomass Burning (BCB)**, based on real-time surface emissions and meteorological data.

This approach transforms **AI climate modeling** from basic numerical solver mimicry to spatio-temporal prediction. By incorporating architectural components that capture trends, seasonal patterns, and cyclical behaviors, the model can deliver accurate long-term forecasts which is essential for multi-decadal to centennial climate projections. This enables higher-resolution, reliable climate simulations without the prohibitive computational costs of full physics parameterizations.
 
***

## Key Features

* **Computational Efficiency:** Significantly reduces the cost of simulating composition transport, enabling higher-resolution and longer transient climate simulations.
* **Interactive Feedback:** Emulates interactive emissions, allowing the climate model to capture real-time feedback between aerosols and other climate processes.
* **Spatiotemporal Architecture:** Utilizes a **ConvLSTM**-based architecture with an inductive bias specifically designed to capture the complex spatial and temporal dependencies in tracer evolution.
* **High Performance:** Achieves excellent performance for BCB concentrations, with **R² values of 0.92** and **Pearson $r$ of 0.96** at the surface level, maintaining acceptable performance even outside the training domain.
* **Focus on Vertical Dynamics:** Incorporates a preprocessing module to effectively fuse 2D emission data with 3D meteorological forcings across the first 20 vertical levels (up to 656 hPa), where most short-term BCB variation occurs.

***

## Model Setup (Conceptual)

The model replaces the full physics solver for tracer transport with an spatiotemporal ML component:
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/nasa-niswan/blob/main/model_setup.png">
  Figure 1. Concept of spatiotemporal ML-based modeling for predicting BCB concentration for climate modeling.
</p>


$$\text{Concentration}(t) = \text{Smart NINT} \left( \text{Emissions}(t), \text{Meteorology}(t), \text{Previous State} \right)$$

### Inputs

* **2D Surface Emissions:** Black Carbon from Biomass Burning (BCB).
* **3D Meteorological Data:** Forcings from ModelE (e.g., pressure, wind, temperature).

### Architecture Highlights

1.  **Preprocessing Module:** Fuses 2D emission data and 3D forcings into a unified spatiotemporal representation.
2.  **Core Model:** A spatiotemporal deep learning model (e.g., ConvLSTM) that learns the complex transport and mixing processes.

***

## Future Work

We aim to extend this methodology to other aerosol and gaseous tracers and integrate the Smart NINT framework into the operational ModelE for comprehensive long-term climate projections.

***

## Reference
Please cite the following papers when referencing this work or any of the foundational concepts discussed, and use the provided links to find the manuscripts:


  * **[Manuscript at NeurIPS 2025](https://arxiv.org/abs/2510.10654)**

    ```bibtex
    @article{erfani2025interactive,
      title={Interactive Atmospheric Composition Emulation for Next-Generation Earth System Models},
      author={Erfani, Seyed Mohammad Hassan and Lamb, Kara and Bauer, Susanne and Tsigaridis, Kostas and van Lier-Walqui, Marcus and Schmidt, Gavin},
      journal={arXiv preprint arXiv:2510.10654},
      year={2025}
    }
    ```

  * **[Manuscript at NeurIPS 2024](https://www.climatechange.ai/papers/neurips2024/66)**

    ```bibtex
    @inproceedings{erfani2024spatiotemporal,
      title={Spatio-Temporal Machine Learning Models for Emulation of Global Atmospheric Composition},
      author={Erfani, Mohammad and Lamb, Kara and Bauer, Susanne and Tsigaridis, Kostas and van Lier-Walqui, Marcus and Schmidt, Gavin},
      booktitle={NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning},
      url={https://www.climatechange.ai/papers/neurips2024/66},
      year={2024}
    }
    ```
