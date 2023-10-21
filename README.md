# Feature Selection for Banking Dataset

This project aims to utilize Genetic Algorithms (GA) to perform feature selection for a bank dataset to enhance the efficiency of predictive models.

## Prerequisites

For this analysis, you need the following Python libraries:

- pandas
- numpy
- sklearn
- lightgbm
- genetic_selection
- matplotlib
- itertools
- DataTable
- warnings

## Data Preprocessing

The bank dataset `bank-full.csv` is loaded and preprocessed. This involves:

- Reading the data into a pandas DataFrame.
- Factorizing categorical data to convert them into numeric form.
- Splitting the data into training and testing datasets.

## Decision Tree

A decision tree classifier is utilized as a baseline model for comparison.

## Genetic Algorithm (GA) for Feature Selection

A Genetic Algorithm approach is utilized to perform feature selection:

- Multiple configurations of the GA parameters such as population size, crossover rate, and mutation rate are tested.
- The best features are selected based on the highest accuracy achieved using cross-validation.
- The results for each configuration are stored, and the best set of features is identified.

## Results Visualization

Various plots are generated:

- A plot to visualize the best fitness value of the GA across generations for different configurations.
- Error plots to understand the standard error for various configurations of the GA.

## Output

The results are stored in a CSV file named `GA_report.csv`.

## Acknowledgment

- Dataset: Moro, Sérgio & Cortez, Paulo & Laureano, Raul. (2012). Enhancing Bank Direct Marketing through Data Mining. CAlg European Marketing Academy. Available at: https://repositorium.sdum.uminho.pt/handle/1822/21409 
- GA Package: Manuel Calzolari. (2021, April 3). manuel-calzolari/sklearn-genetic: sklearn-genetic 0.4.1 (Version 0.4.1). Zenodo. http://doi.org/10.5281/zenodo.4661248
