# BiXiao Model — v1.0: Minimal Executable Demonstration Package

The BeXiaoCastNet3D.ts and BiXiaoCastNet3DENV.ts should be downloaded firstly in Zenodo ([![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17591610.svg)](https://doi.org/10.5281/zenodo.17591610))
This package provides a minimal executable demonstration of the **BiXiao atmospheric environmental forecasting model**.  
The input data required for driving the meteorological module and the environmental module are stored in the directories `input_met` and `input_env`, respectively.

The package also includes a plotting script for comparing the model forecasts with observational data.  
The observational data used for visualization are located in the `envData_for_plots` directory.

By default, the model uses **NVIDIA CUDA** for accelerated computation. CPU-only inference is also supported, although the runtime will be significantly longer.

---

## Prerequisites

It is recommended to use **Anaconda** to configure the Python environment.  
The required software versions are:

- Python 3.10  
- PyTorch 2.9.0  
- CUDA (recommended: 13.0)  
- NumPy 2.2.5  
- Xarray 2025.6.1  
- NetCDF library 4.9.3  
- Matplotlib 3.10.5  

---

## Usage Instructions

1. Place all contents of this package in an appropriate working directory.  
2. Ensure that your current working directory is the root of this package, as the scripts use relative paths.  
   Alternatively, you may modify the paths in the scripts to absolute paths if needed.  
3. Run `python inference_met.py`.  
   (To switch to CPU inference, replace `"cuda"` with `"cpu"` in the script.)  
4. Run `python inference_env.py`.  
   (Similarly, replace `"cuda"` with `"cpu"` if CPU-only inference is desired.)  
5. The resulting meteorological and environmental forecasts will be saved in the directories `output_met` and `output_env`.  
6. To visualize results, run `visualize.py`. The observational data in `envData_for_plots` will be used for comparison.

---

## Minimal Sample Data Description

The input files provided in `input_met` and `input_env` correspond to the **heavy PM₂.₅ pollution episode analyzed in the manuscript**, selected as a representative demonstration case.  
This episode allows the user to reproduce the complete end-to-end inference workflow and validate the model’s behavior under a typical severe pollution scenario.

The observational surface air-quality data used for visualization are located in the `envData_for_plots` directory and correspond to the same pollution event. These data are used to generate comparison figures through `visualize.py`.

---

## Important Note

During model training, **30 grid points** were included.  
However, the **9th grid point (Python index = 8)** contains substantial missing data across the training period, making it unsuitable for meaningful analysis.  
Therefore, only **29 grid points** should be regarded as valid.

---

## License

This project is distributed under the  
**GNU Affero General Public License v3.0 only (AGPL-3.0-only)**.

Key points of this license include:

- You may use, modify, and redistribute the code **freely**.  
- Any modified versions or derivative works **must also be released under AGPL-3.0-only**.  
- If you make the software available to users over a **network or web service**, the complete corresponding source code — including your modifications — **must be provided**.  
- The license ensures that improvements to the software remain **open and accessible** to the community.
