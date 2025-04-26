Implementation of the paper "Federated Learning for Critical Electrical Infrastructure - Handling Non-IID Data for Predictive Maintenance of Substation Equipment" by Soham Ghosh and Gaurav Mittal.

There are three datasets that are provided in this repository:

1. HV Circuit Breaker Maintenance Data (Excel file)

    -   The dataset is derived by combining results from high-voltage breaker monitoring at utility substations, where sensors capture critical measurements such as SF6 density, breaker status, and ambient cabinet temperature.
    -   Utility engineers at participating utilities routinely review breaker records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as SF6 dew point, fault operation count, contact life, and days since last operation, in accordance with each utility’s established maintenance standards.
    -   The dataset comprises 5,000 samples of breaker readings, aggregated from five representative utilities. Each breaker is assigned a unique categorical identifier corresponding to its source utility, with 28 features representing various sensor-based measurements and one quality metric (0: no maintenance required, 1: maintenance required).
    -   It is important to note that not all 28 features are uniformly available across all product variants, leading to slight variations in the feature space between variants.
    -   To enhance the training of machine learning models, the dataset has been augmented with an increased proportion of negative examples to ensure sufficient representation of failure cases.

2. Large Power Transformer Maintenance Data (Excel file)

    -   The dataset is similar to the HV Circuit Breaker Maintenance Data Set, and is derived by combining results from large power transformer monitoring at utility substations, where sensors capture critical measurements such as LTC and main tank oil temperature, dissolved gas values (in ppm).
    -   Utility engineers at participating utilities routinely review transformer records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as high oil temperature, and excessive amount of certain dissolved gas (usually based on Duval triangles and pentagons), following each utility’s established maintenance standards.


3. Emergency Station Generator Maintenance Data (Excel file)

 -   The dataset is derived by combining results from station emergency generators at utility substations (~150 - 200 kVA on propane or natural gas), where sensors capture critical measurements such as engine temperatures, oil pressure, alternate current and voltages.
 Utility engineers at participating utilities routinely review station emergency generator records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as start attempts, battery state of charge, lube oil temperature, crankcase pressure, and emission levels, in accordance with each utility’s established maintenance standards.

Additional notes: 

1. The 'Federated Learning for Breaker Main.ipynb' notebook uses the 'HV Circuit Breaker Maintenance Data' dataset and contains Python code for the implementation of IID and non-IID comparisons, along with performance observations for FedAvg, FedProx, and FedBN.
2. The 'Federated Learning for Breaker Advanced.ipynb' notebook uses the same 'HV Circuit Breaker Maintenance Data' dataset and contains additional Python code for implementation of non-IID data partitions  with performance comparison between FedAvg, FedAvg with Momentum, FedProx, and FedBN.

Disclaimer:
The databases are provided "as is" without any express or implied warranties, including, but not limited to, implied warranties of merchantability and fitness for a particular purpose. In no event shall we be liable for any direct, indirect, incidental, special, exemplary, or consequential damages, however caused and under any theory of liability, whether in contract, strict liability, tort (including negligence or otherwise), or otherwise, arising in any way out of the use of the database. By downloading, accessing, or using the database, you hereby release us from any and all liability that may arise from your use of the database. 

These databases are intended for educational and research purposes only and should not be used as a substitute for real-world electrical asset protection, operational decision-making, or maintenance planning.
