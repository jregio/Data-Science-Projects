# Data-Science-Projects
A collection of data science projects to help me practice working with data across diverse domains
---

**Anomaly Detection for Financial Data (01/26/26)**

Binary classification competition for anomaly detection in VDI process financial data. Dataset contained binary-encoded features from financial transactions requiring classification of anomalous vs. normal samples.

Systematically evaluated three modeling approaches of decreasing complexity:
1. PCA dimensionality reduction + Random Forest (F1: 22%)
2. Random Forest with hyperparameter tuning (F1: 57%)
3. Logistic Regression with default parameters (F1: 100%)

The perfect F1-score achieved with the simplest model demonstrated an important ML principle: always establish baseline performance with simple models before adding complexity. The linearly separable nature of the binary-encoded data made logistic regression the optimal choice, outperforming more sophisticated ensemble methods.

This project reinforced the value of systematic model selection and the importance of matching model complexity to problem complexity.

[Link to competition](https://www.kaggle.com/competitions/anomaly-dectection-for-financial)
