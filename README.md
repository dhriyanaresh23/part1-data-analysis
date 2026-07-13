# Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

## Project Overview

This project focuses on data acquisition, preprocessing, cleaning, and exploratory data analysis (EDA) using the House Prices dataset. 
The objective is to prepare a high-quality dataset for machine learning by identifying and handling missing values, correcting data types, detecting duplicate records, analyzing statistical properties, visualizing relationships, performing correlation analysis, and exporting a cleaned dataset for subsequent parts of the capstone project.

The notebook follows a structured workflow that includes data loading, cleaning, exploratory analysis, visualization, statistical comparison, and dataset export.

---
# Objectives

The main objectives of this project are:

- Load and inspect the House Prices dataset.
- Identify and handle missing values.
- Remove duplicate records.
- Correct inappropriate data types.
- Reduce memory usage where possible.
- Perform exploratory data analysis.
- Detect skewness and outliers.
- Create multiple visualizations.
- Perform correlation analysis.
- Compare Pearson and Spearman correlations.
- Generate grouped statistical summaries.
- Export a cleaned dataset for future machine learning tasks.

---
# Dataset Information

**Dataset Name:** House Prices Dataset

Dataset Dimensions:

- Rows: **1460**
- Columns: **81**

The dataset contains both numerical and categorical variables describing residential properties.

Important attributes include:

- Lot Area
- Overall Quality
- Year Built
- Garage Features
- Basement Features
- Living Area
- Sale Price

The target variable used in later parts is:

**SalePrice**

---

# Repository Structure

```
part1-data-analysis/
│
├── part1_analysis.ipynb
├── cleaned_data.csv
├── README.md
│
└── outputs/
    ├── line_plot.png
    ├── bar_chart.png
    ├── histogram.png
    ├── scatter_plot.png
    ├── box_plot.png
    └── correlation_heatmap.png
```

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

---
# Dependencies

Install the required libraries before running the notebook.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

---
# How to Run

## Step 1

Clone the repository.

```bash
git clone https://github.com/dhriyanaresh23/part1-data-analysis.git
```

## Step 2

Open the project folder.

```bash
cd part1-data-analysis
```

## Step 3

Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Step 4

Launch Jupyter Notebook.

```bash
jupyter notebook
```

## Step 5

Open:

```text
part1_analysis.ipynb
```

## Step 6

Run all notebook cells from top to bottom.

The notebook automatically generates:
- `cleaned_data.csv`
- Visualization images
- Correlation matrix
- Statistical summaries

---

# Environment Variables

This project **does not require any API keys or environment variables**.

No passwords, API keys, tokens, or secrets are stored anywhere in this repository.

---

# Task 1 – Load the Dataset

## Objective

Load the House Prices dataset into a Pandas DataFrame and inspect its structure.

## Implementation

The dataset was loaded using:

```python
pd.read_csv()
```

The following information was displayed:

- First five rows
- Dataset shape
- Column names
- Data types

## Result

Dataset Shape:

- **1460 rows**
- **81 columns**

The dataset consists of numerical and categorical variables that describe different characteristics of residential properties.

The initial inspection confirmed that several columns contained missing values and required preprocessing before further analysis.

---

# Task 2 – Null Value Analysis

## Objective

Identify missing values and perform appropriate imputation.

## Implementation

Missing values were identified using:

```python
df.isnull().sum()
```

The percentage of missing values for each column was calculated using:

```python
(df.isnull().sum() / df.shape[0]) * 100
```


A null-value summary table was generated to display both the count and percentage of missing values for every column.

Columns with a missing-value percentage greater than **20%** were identified and reported separately.

For numeric columns with **less than 20% missing values**, missing values were filled using the column median:

```python
df[col].fillna(df[col].median(), inplace=True)
```

## Why Median Instead of Mean?

Median imputation was selected because many housing-related variables are positively skewed and contain extreme values.

Unlike the mean, the median is resistant to outliers and therefore preserves the natural distribution of the data.

This approach reduces bias introduced by unusually large or expensive properties.

---
## Results

- Missing-value counts and percentages were calculated for every column.
- Columns exceeding a **20% null rate** were identified and reported.
- Numeric columns with less than **20% missing values** were successfully imputed using the median.
- After imputation, the dataset contained significantly fewer missing values and was ready for further preprocessing.

# Task 3 – Duplicate Detection and Removal

## Objective

Identify duplicate observations and remove them if necessary.

## Implementation

Duplicate rows were detected using:

```python
df.duplicated().sum()
```

Duplicate records were removed using:

```python
df.drop_duplicates()
```

## Result

Duplicate Rows Found:

**0**

Duplicate Rows Removed:

**0**

Since no duplicate observations were present, the dataset size remained unchanged after this step.

This confirmed that every observation represented a unique property record.

# Task 4: Data Type Correction

## Objective

The objective of this task was to identify columns with incorrect data types, convert them to appropriate data types, and reduce memory usage by converting repetitive string columns to the `category` data type.

## Methodology

The data types of all columns were inspected using:

```python
df.dtypes
```

Columns with incorrect inferred data types were converted using either:

```python
astype()
```

or

```python
pd.to_numeric(errors="coerce")
```

where appropriate.

At least one repetitive string column was converted to the `category` data type to improve memory efficiency.

Memory usage before and after the conversion was calculated using:

```python
df.memory_usage(deep=True).sum()
```

## Results

- Incorrect data types were successfully corrected.
- At least one categorical column was converted to the `category` data type.
- Memory usage before and after conversion was measured.
- The dataset became more memory efficient while preserving the original information.

## Design Decision

Using appropriate data types improves computational efficiency, reduces memory consumption, and prepares the dataset for efficient machine learning preprocessing.

---

# Task 5: Descriptive Statistics and Skewness

## Objective

The objective of this task was to generate descriptive statistics for all numerical features, measure skewness, identify the most skewed feature, and explain how skewness influences the choice of imputation strategy.

## Methodology

Descriptive statistics were generated using:

```python
df.describe()
```

Skewness for each numerical column was calculated using:

```python
df[col].skew()
```

The column with the highest absolute skewness was identified.

After comparing the skewness values of all numerical columns, **MiscVal** was identified as the column with the highest absolute skewness.

- **Highest Absolute Skewness Column:** MiscVal
- **Skewness Value:** 24.476794
- 
## Results

The descriptive statistics included:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile (Q1)
- Median (50%)
- 75th percentile (Q3)
- Maximum

After comparing the skewness values of all numerical columns, **MiscVal** was identified as the column with the highest absolute skewness.

### Interpretation

The **MiscVal** column represents the monetary value of miscellaneous property features that are not covered by other attributes. Most houses have a value of **0**, while only a few properties have very high miscellaneous values. This creates a long right tail in the distribution, resulting in a highly **positively skewed** distribution.

A **positive skew** means that a small number of extremely large values pull the distribution toward the right.

A **negative skew** means that a small number of extremely small values pull the distribution toward the left.

### Effect on Missing Value Imputation

For highly skewed features such as **MiscVal**, the **mean** is affected by extreme values and therefore does not accurately represent the typical observation.

The **median** is resistant to outliers and skewed distributions because it is based on the middle value rather than the average. Therefore, the median is a more appropriate choice for imputing missing values in skewed numerical features.

# Task 6: Outlier Detection with IQR

## Objective

The objective of this task was to identify potential outliers in the dataset using the Interquartile Range (IQR) method without removing any observations. The identified outliers will be considered during model development in Part 2.

## Methodology

Outlier analysis was performed on the following numerical columns:

- SalePrice
- GrLivArea

For each column, the following statistics were calculated:

- First Quartile (Q1)
- Third Quartile (Q3)
- Interquartile Range (IQR)

The IQR was computed using:

```text
IQR = Q3 − Q1
```

The lower and upper bounds were calculated as:

```text
Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Any observation lying outside these bounds was considered an outlier.

## Results

### SalePrice

| Statistic | Value |
|-----------|-------:|
| Q1 | 129975.00 |
| Q3 | 214000.00 |
| IQR | 84025.00 |
| Lower Bound | 3937.50 |
| Upper Bound | 340037.50 |
| Outlier Count | **61** |

### GrLivArea

| Statistic | Value |
|-----------|-------:|
| Q1 | 1129.50 |
| Q3 | 1776.75 |
| IQR | 647.25 |
| Lower Bound | 158.625 |
| Upper Bound | 2747.625 |
| Outlier Count | **31** |

Outliers were identified for the selected numerical features.

The number of observations outside the IQR boundaries was calculated and reported.

No outliers were removed during this stage because the assignment specifically required only their identification and documentation.

## Interpretation

The IQR analysis identified **61 outliers** in the **SalePrice** feature and **31 outliers** in the **GrLivArea** feature.

These outliers are likely to represent genuinely expensive properties or unusually large houses rather than data entry errors. Removing them at this stage could eliminate important information from the dataset and reduce the model's ability to learn real-world variation.

Therefore, the identified outliers were **retained** in the cleaned dataset. If required during model development, robust machine learning algorithms or feature transformation techniques can be applied instead of deleting these observations.

# Task 7: Data Visualizations

## Objective

The objective of this task was to explore the cleaned dataset visually, understand the distribution of important variables, identify possible outliers, examine relationships between features, and detect correlations that may be useful during model development.

---

Five different visualizations were created to better understand the cleaned housing dataset. Each visualization helps identify patterns, relationships, distributions, and potential outliers that can support future predictive modelling.

---

## 1. Line Plot

**Plot Title:** SalePrice Across Dataset

A line plot was created for the **SalePrice** column using the dataset row index.

### Interpretation

The line plot shows that house prices fluctuate throughout the dataset. Most properties are clustered within a moderate price range, while a few observations have exceptionally high sale prices. These peaks indicate the presence of expensive properties that may be treated as outliers during analysis.

---

## 2. Bar Chart

**Plot Title:** Mean SalePrice by Neighborhood

A bar chart was created to compare the average **SalePrice** across different **Neighborhood** categories.

The average sale price for each neighborhood was computed using group-by aggregation and displayed as a bar chart.


### Interpretation

The bar chart shows noticeable differences in average sale prices across neighborhoods. Neighborhoods such as NoRidge have considerably  higher average sale prices than others, indicating that location has a strong influence on property values. This suggests that the Neighborhood feature is likely to be an important predictor of house prices in the dataset

## 3. Histogram

**Plot Title:** Distribution of MiscVal

A histogram with **20 bins** was created for the **MiscVal** column using `sns.histplot()`, which had the highest absolute skewness.

### Shape of the Distribution

The histogram has a **highly right-skewed (positively skewed)** distribution. Most observations are concentrated at **0**, creating a tall peak on the left side of the histogram, while only a small number of observations have very large miscellaneous values. These extreme values create a **long right tail**, showing that the data is not normally distributed.

### Interpretation

The histogram confirms that **MiscVal** is the most skewed numerical feature. Because of its heavy positive skew and extreme values, the **median** is a better choice than the mean for imputing missing values.


## 4. Scatter Plot

**Plot Title:** GrLivArea vs SalePrice

A scatter plot was created using `sns.scatterplot()` to visualize the relationship between **GrLivArea** and **SalePrice**.

### Interpretation

The scatter plot shows a **positive and moderately strong linear relationship** between **GrLivArea** and **SalePrice**. As the above-ground living area increases, the sale price generally increases as well. Although a few outliers are present, the overall upward trend indicates that larger houses tend to have higher selling prices.


## 5. Box Plot

**Plot Title:** SalePrice Distribution by HouseStyle

A box plot was created to compare the distribution of **SalePrice** across different **HouseStyle** categories.

### Interpretation

The box plot shows clear differences in the **median sale price** and the **spread of prices** across different house styles. Some categories have higher median sale prices, while others exhibit greater variability and more outliers. This suggests that house style influences the selling price and contributes useful information for predictive modelling.


## Summary

The visualizations provided several important insights:

- SalePrice varies considerably across the dataset.
- Neighborhood has a significant impact on average selling price.
- MiscVal has a highly right-skewed distribution with many zero values and a few extreme observations.
- GrLivArea and SalePrice have a positive, moderately strong relationship.
- HouseStyle influences both the median and variability of sale prices.
- Outliers are present in multiple visualizations, but they appear to represent genuine property characteristics rather than data-entry errors.

# Task 8: Correlation Heat Map

## Objective

The objective of this task was to identify relationships between numerical variables by computing a correlation matrix and visualising it using a heat map. This helps identify strongly related features that may be useful during feature selection.

## Methodology

A correlation matrix was computed using:

```python
df.corr()
```

The resulting matrix was visualised using a heat map with annotations to display the correlation values between numerical features.

## Highest Correlation Pair

The pair with the highest absolute correlation was:

| Variable 1 | Variable 2 | Correlation |
|------------|------------|------------:|
| GarageCars | GarageArea | **0.882475** |

## Interpretation

The strong positive correlation between **GarageCars** and **GarageArea** indicates that houses with larger garage areas generally accommodate more vehicles. This relationship is expected because both variables describe the size and capacity of the garage.

This correlation does **not** necessarily imply a causal relationship. Instead, both variables are likely influenced by the overall size and design of the property.

A plausible alternative explanation is that **larger houses tend to have both larger garages and greater garage capacity**, causing these two variables to increase together.

The correlation heat map helps identify highly related features and provides useful guidance for feature selection in later modelling stages.

# Task 9(a): Imputation Strategy Comparison

## Objective

The objective of this task was to compare the **mean** and **median** for the two numerical columns with the highest absolute skewness before applying missing value imputation. This comparison helps determine the most appropriate measure of central tendency for skewed data.

## Columns Analysed

The two numerical columns with the highest absolute skewness identified in **Task 5** were:

- MiscVal
- PoolArea

## Mean and Median Comparison

| Column | Skewness | Mean | Median |
|---------|---------:|-----:|-------:|
| MiscVal | 24.476794 | 43.489041 | 0.0 |
| PoolArea | 14.828374 | 2.758904 | 0.0 |

## Imputation Strategy

Both **MiscVal** and **PoolArea** are **positively skewed**.

For a **positively skewed** distribution, a small number of extremely high values pull the **mean upward**, making it larger than the value representing most observations. The **median** is less affected by these extreme values and therefore provides a more representative measure of central tendency.

For a **negatively skewed** distribution, a small number of extremely low values pull the **mean downward**, while the median remains more stable and representative of the centre of the distribution.

Since both selected columns are highly **positively skewed**, the **median** was chosen for imputing missing values using `fillna()`.

## Missing Value Confirmation

After applying median imputation, the remaining missing values were checked using `isnull().sum()`.

| Column | Remaining Null Values |
|---------|----------------------:|
| MiscVal | 0 |
| PoolArea | 0 |

The results confirm that **no missing values remain** in either column after imputation.

# Task 9(b): Spearman Rank Correlation

## Objective

The objective of this task was to compare Pearson and Spearman correlation coefficients to determine whether relationships between numerical variables are approximately linear or monotonic but non-linear. This comparison helps identify the most suitable correlation measure for feature selection in the modelling stage.

## Three Variable Pairs with the Largest Difference

| Variable Pair | Pearson | Spearman | |Spearman − Pearson| |
|---------------|---------:|----------:|---------------------:|
| LotFrontage – LotArea | 0.304522 | 0.554082 | 0.249559 |
| LotArea – BedroomAbvGr | 0.119690 | 0.337788 | 0.218098 |
| LotArea – TotRmsAbvGrd | 0.190015 | 0.405924 | 0.215909 |

## Interpretation

### 1. LotFrontage and LotArea

Since **|Spearman| (0.554082) > |Pearson| (0.304522)**, this relationship appears to be **monotonic but non-linear**. As the lot area increases, the lot frontage generally increases as well, but the relationship is not perfectly linear.

### 2. LotArea and BedroomAbvGr

Since **|Spearman| (0.337788) > |Pearson| (0.119690)**, this relationship is also **monotonic but non-linear**. Larger lots generally contain houses with more bedrooms, although the increase is not proportional.

### 3. LotArea and TotRmsAbvGrd

Since **|Spearman| (0.405924) > |Pearson| (0.190015)**, this relationship is **monotonic but non-linear**. Properties with larger lot areas generally have more rooms above ground, but the relationship is not strictly linear.

## Correlation Measure for Feature Selection

For feature selection in **Part 2**, **Spearman correlation** will be preferred because it captures monotonic relationships that may not be strictly linear. Since all three variable pairs have **|Spearman| greater than |Pearson|**, Spearman provides a stronger and more representative measure of association for these features. Using Spearman correlation can help identify useful predictors even when relationships are non-linear.

# Task 9(c): Grouped Aggregation

## Objective

The objective of this task was to analyse how the categorical feature **Neighborhood** influences the numerical target variable **SalePrice** by computing grouped summary statistics.

The dataset was grouped using:

```python
df.groupby("Neighborhood")["SalePrice"].agg(["mean", "std", "count"])
```

The mean, standard deviation, and count were calculated for each neighbourhood.

## Results

### Group with the Highest Mean

- **Neighborhood:** NoRidge
- **Mean SalePrice:** 335295.32

### Group with the Highest Standard Deviation

- **Neighborhood:** NoRidge
- **Standard Deviation:** 121412.66

### Ratio of Highest Mean to Lowest Mean

| Highest Mean | Lowest Mean | Ratio |
|--------------|------------:|------:|
| 335295.32 | 98576.47 | **3.40** |

## Interpretation

The **NoRidge** neighbourhood has both the highest average sale price and the highest variation in sale prices.

The high within-group standard deviation indicates that house prices vary considerably even within the same neighbourhood. This is a concern for a predictive model because the **Neighborhood** feature alone is not sufficient to predict the sale price reliably for every property in that group. Other features such as house size, overall quality, living area, garage size, and year built also contribute to the final sale price.

The ratio of the highest neighbourhood mean to the lowest neighbourhood mean is **3.40**, indicating that the average sale price in the most expensive neighbourhood is more than three times higher than that of the least expensive neighbourhood.

This large ratio suggests that **Neighborhood carries useful predictive signal** and should be retained as an important feature for modelling in Part 2.

# Task 10: Saving the Cleaned Dataset

After completing data cleaning, preprocessing, and exploratory data analysis, the final cleaned dataset was saved as:

```text
cleaned_data.csv
```

using:

```python
df.to_csv("cleaned_data.csv", index=False)
```

The cleaned dataset is included in this repository and serves as the input dataset for Parts 2 and 3 of the project.

The file contains the cleaned records after:

- Handling missing values
- Removing duplicate records
- Correcting data types
- Performing exploratory data analysis
- Preparing the dataset for predictive modelling

# Conclusion

This project successfully completed data acquisition, preprocessing, cleaning, exploratory data analysis, visualization, correlation analysis, skewness analysis, outlier detection, and grouped statistical analysis for the House Prices dataset.

The cleaned dataset (`cleaned_data.csv`) was generated and prepared for subsequent machine learning tasks in Parts 2 and 3. The analysis identified important predictors such as Neighborhood, GrLivArea, GarageArea, and OverallQual while preserving genuine outliers for future modelling.
