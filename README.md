# classical_ml_project

# Project Name

Short project description here.

## Table of Contents

- [Tech Stack Used](#tech-stack-used)
- [Feature Selection and Imputation](#feature-selection-and-imputation)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Various Models Used](#various-models-used)
- [Final Technique: Stacking](#final-technique-stacking)
- [Final Results](#final-results)
- [References](#references)

## Tech Stack Used

- **Programming Language**: Python 3.x
- **Libraries**: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - [Any other libraries or frameworks used]

## Feature Selection and Imputation

### Feature Selection Steps:

1. **Handling Missing Data**: The project initially addressed missing data by dropping rows with NaN values, taking into consideration that nan values for each feature were less than 2%. This ensured that the remaining data was largely complete.

2. **Feature Importance Analysis**: A feature importance analysis was conducted on the dataset using techniques like Random Forest Feature Importance or Gradient Boosting Feature Importance. This analysis helped identify the most relevant features for modeling.

3. **Feature Removal**: Based on the feature importance analysis, less relevant or redundant feature columns were removed to improve the efficiency of the model and reduce the dimensionality of the dataset. The following features where dropped based on its over all feature importance. ['Source', 'Month']

![Feature Importance Analysis](images/feature_importance_lgbm.png)

### Feature Imputation Steps:

1. **Categorical Data Handling**: Categorical data was handled by applying one-hot encoding to convert categorical variables into numerical format.

2. **Data Distribution Analysis**: Density graphs were plotted for each feature to analyze their data distributions.

3. **Imputation Based on Data Distribution**: Univariate imputation techniques were used due to the data characteristics and the limited number of missing values. Specifically:
   - For features with a normal distribution, missing values were imputed using the mean of that feature.
   - For features with skewed distributions, missing values were imputed using the median of that feature.

4. **Columns with Imputed Values**: Below is the list of columns where imputation was performed, along with the method used for each:
   - Mean Imputation:
     - ['pH','Air Temperature']
   - Median Imputation:
     - ['Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc' , 'Turbidity', 'Fluoride', 'Copper', 'Total Dissolved Solids', 'Water Temperature','Chlorine', 'Conductivity', 'Sulfate', 'Odor', 'Manganese', 'Day', 'Time of Day']

The decision to use univariate imputation techniques like mean and median was made based on the data characteristics and the small number of missing values, which eliminated the need for more advanced techniques such as MICE.



## Synthetic Data Generation

### Synthetic Data Using CTGAN Model

To address the challenge of a highly imbalanced dataset, synthetic data was generated using the CTGAN (Conditional Tabular GAN) model. CTGAN is a state-of-the-art generative adversarial network (GAN) specifically designed for generating synthetic tabular data. It learns the underlying data distribution and generates synthetic samples that closely resemble the original data while preserving statistical properties.

CTGAN was chosen for synthetic data generation due to its ability to:

- Balance the class distribution in the dataset, effectively addressing class imbalance issues.
- Generate synthetic data that captures the intricate relationships between features present in the original dataset.

### Use Case for Synthetic Data

The need for synthetic data arose from the substantial class imbalance in the original dataset. By generating synthetic samples with CTGAN, we balanced the class distribution, creating a more representative and balanced dataset. This balanced dataset was then used for training machine learning models, improving their performance and generalization.

By leveraging synthetic data, we aimed to enhance the model's ability to make accurate predictions on underrepresented classes, ultimately improving the overall model performance and fairness.


## Various Models Used

| Model Name                  | Time (with scikit-learn-intelex)      | Time (without scikit-learn-intelex) | 
|-----------------------------|---------------------------------------|-------------------------------------|
| RandomForestClassifier      | 3.2 milliseconds                      | 19.4 milliseconds                   |
| BaggingClassifier           | 4.2 milliseconds                      | 3.53 milliseconds                   |
| AdaboostClassifier          | 20.9 milliseconds                     | 32.2 milliseconds                   |
| GradientBoostingClassifier  | 4.1 milliseconds                      | 1.39 milliseconds                   |

  
## Final Technique: Stacking

Explain the final technique used, such as model stacking, ensemble methods, or any other advanced techniques. Provide details on how these models were combined and why this approach was chosen.

## Final Results

Share the key results and metrics obtained from your models. This could include accuracy, precision, recall, F1-score, or any domain-specific metrics. You can also include visualizations like confusion matrices or ROC curves to illustrate your results.

## References

List any references, research papers, or libraries you used in your project. Properly cite any external sources to give credit to the authors and maintain good academic practice.

