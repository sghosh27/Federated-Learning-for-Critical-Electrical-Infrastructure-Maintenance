# Federated Learning for Critical Electrical Infrastructure

**Authors**: Soham Ghosh and Gaurav Mittal

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
  - [HV Circuit Breaker Maintenance Data](#1-hv-circuit-breaker-maintenance-data)
  - [Large Power Transformer Maintenance Data](#2-large-power-transformer-maintenance-data)
  - [Emergency Station Generator Maintenance Data](#3-emergency-station-generator-maintenance-data)
3. [Notebooks](#notebooks)
  - [Federated_Learning_for_Breaker_Main.ipynb](#federated_learning_for_breaker_mainipynb)
  - [Federated Learning for Breaker Advanced.ipynb](#federated-learning-for-breaker-advancedipynb)
  - [Additional Notebooks](#additional-notebooks)
4. [Running Jupyter Notebooks in Google Colab](#running-jupyter-notebooks-in-google-colab)
  - [Running Federated_Learning_for_Breaker_Main.ipynb](#running-federated_learning_for_breaker_mainipynb)
5. [Running Jupyter Notebooks in Other Environments](#running-jupyter-notebooks-in-other-environments)
6. [Disclaimer](#disclaimer)


## Introduction

This repository contains the implementation of the paper **"Federated Learning for Critical Electrical Infrastructure - Handling Non-IID Data for Predictive Maintenance of Substation Equipment"**.


## Datasets

### 1. HV Circuit Breaker Maintenance Data

- **Description**:  
  The dataset is derived by combining results from high-voltage breaker monitoring at utility substations, where sensors capture critical measurements such as SF6 density, breaker status, and ambient cabinet temperature.  
  *Note*: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.

- **Details**:
  - Utility engineers routinely review breaker records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These flags are typically based on:
    - SF6 dew point
    - Fault operation count
    - Contact life
    - Days since last operation  
    (in accordance with each utility’s established maintenance standards).
  - The dataset comprises **5,000 samples** of breaker readings aggregated from five representative utilities.
  - Each breaker is assigned a unique categorical identifier corresponding to its source utility.
  - **Features**: 28 sensor-based measurements and one quality metric:
    - `0`: No maintenance required
    - `1`: Maintenance required
  - Not all 28 features are uniformly available across all product variants, leading to slight variations in the feature space.
  - The dataset has been augmented with an increased proportion of negative examples to ensure sufficient representation of failure cases.


### 2. Large Power Transformer Maintenance Data

- **Description**:  
  Similar to the HV Circuit Breaker Maintenance Data, this dataset is derived by combining results from large power transformer monitoring at utility substations, where sensors capture critical measurements such as LTC and main tank oil temperature, and dissolved gas values (in ppm).  
  *Note*: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.

- **Details**:
  - Utility engineers routinely review transformer records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These flags are typically based on:
    - High oil temperature
    - Excessive amounts of certain dissolved gases (usually based on Duval triangles and pentagons)  
    (in accordance with each utility’s established maintenance standards).


### 3. Emergency Station Generator Maintenance Data

- **Description**:  
  This dataset is derived by combining results from station emergency generators at utility substations (~150–200 kVA on propane or natural gas), where sensors capture critical measurements such as engine temperatures, oil pressure, alternate current, and voltages.  
  *Note*: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.

- **Details**:
  - Utility engineers routinely review station emergency generator records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These flags are typically based on:
    - Start attempts
    - Battery state of charge
    - Lube oil temperature
    - Crankcase pressure
    - Emission levels  
    (in accordance with each utility’s established maintenance standards).


## Notebooks

1. **`Federated_Learning_for_Breaker_Main.ipynb`**:
  - Utilizes the **HV Circuit Breaker Maintenance Data** Excel file for in-depth analysis.
  - Executes models to evaluate and compare **Training Loss** and **Training Accuracy** for both IID and non-IID data partitions using the **FedAvg** method.
  - Extends model execution to compare **Training Loss** and **Training Accuracy** for IID data using **FedAvg** and for non-IID data using **FedProx**.
  - Offers performance insights and observations for **FedAvg**, **FedProx**, and **FedBN** under IID partitioning scenarios.
  - Includes comparisons of Label Skew and Feature Skew for **FedAvg**, **FedProx**, and **FedBN** under non-IID partitioning conditions.

2. **`Federated Learning for Breaker Advanced.ipynb`**:
   - Uses the **HV_Circuit_Breaker_Maintenance_Data** dataset excel file.
   - Contains additional Python code for the implementation of non-IID data partitions.
   - Includes performance comparisons between FedAvg, FedAvg with Momentum, FedProx, and FedBN.

3. **Additional Notebooks**:
   - **`Federated Learning for Transformer.ipynb`**: Uses the **Large Power Transformer Maintenance Data** dataset.
   - **`Federated Learning for Generator.ipynb`**: Uses the **Emergency Station Generator Maintenance Data** dataset.


## Running Jupyter Notebooks in Google Colab

For seamless execution of the Jupyter notebooks, we recommend using the Google Colab environment. This ensures compatibility with the required dependencies and avoids versioning issues. Use the badge below to open the notebooks directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance/)


### Running **`Federated_Learning_for_Breaker_Main.ipynb`**

1. **Open the Notebook**:  
  Launch **`Federated_Learning_for_Breaker_Main.ipynb`** in the Google Colab environment.

2. **Upload Required Files**:  
  - Use the file upload option in Colab to upload the necessary files.  
  - Click the folder icon on the left sidebar to open the file explorer.  
  - Use the upload button (shaped like a file with an upward arrow) to upload the following files from the GitHub repository:
    - **`HV_Circuit_Breaker_Maintenance_Data.xlsx`** (located in the `Datasets` folder).
    - **`breaker_functions.py`** (located in the `Notebooks` folder).

3. **Install Dependencies**:  
  - Run Step 1 in the notebook to install the required packages.  
  - If prompted, restart the session after the installation completes.

4. **Execute the Notebook**:  
  - Run the entire notebook or execute each cell sequentially to view the outputs and analyses.

4. **Run the Notebook**:  
  - Execute the cells in the notebook sequentially to reproduce the results and analyses.



## Running Jupyter Notebooks in Other Environments
If you intend to run these notebooks in an environment other than Google Colab, you can utilize the `requirements.txt` file extracted from the Colab environment. This file provides a starting point with a list of necessary packages to set up your environment. However, we cannot guarantee that all packages and versions will be fully compatible across different environments. You may need to make adjustments and modifications to the package list and versions specified in the `requirements.txt` file. The file is provided strictly as a reference to assist in identifying the dependencies required to run these notebooks.

To install all the dependencies listed in the `requirements.txt` file into your environment, use the following command:

```bash
pip install -r requirements.txt
```


$\textcolor{red}{DISCLAIMER:}$
These datasets are semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data. The databases are provided "as is" without any express or implied warranties, including, but not limited to, implied warranties of merchantability and fitness for a particular purpose. In no event shall we be liable for any direct, indirect, incidental, special, exemplary, or consequential damages, however caused and under any theory of liability, whether in contract, strict liability, tort (including negligence or otherwise), or otherwise, arising in any way out of the use of the database. By downloading, accessing, or using the database, you hereby release us from any and all liability that may arise from your use of the database. 

Given that these databases are intended for educational and research purposes only, it is imperative that these are not used as a substitute for real-world electrical asset protection, operational decision-making, or maintenance planning.