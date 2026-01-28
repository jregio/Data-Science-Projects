# Data-Science-Projects
A collection of data science projects to help me practice working with data across diverse domains

## Computer Vision for Jaguar Re-identification

### Objective
**Jaguar identification** identifies an individal jaguar based on the physical features. The goal of this project is to use computer vision to predict if two pictures of a jaguar represent the same individual.

### Data 
The training data consists of 1895 images of jaguars labelled with the name of the individual jaguar. The test data consists of 371 unlabelled images of jaguars.

### What I did 
The pipeline consisted of image augmentations followed by using Meta's foundational computer vision model **DINOv2** to extract features from the images and embed them into vectors. The **MPerClassSampler** was used during training to handle class imbalances in the dataset. The **ArcFace** loss was used to cluster learned embeddings into tight clusters in order to better calculate similarity scores. After training, learned embeddings were computed on the test set and cosine similarity scores were calculated for each pair of images. The pipeline was run on cloud GPU computing using **Vast.ai**.

### Results 
Initially, the model was trained for 72 epochs using **DINOv2_small** and achieved a score of 75.3% mAP score. After switching to **DINOv2** and training for 200 epochs, the model achieved a mAP score of 78.5%.

### What I learned
I learned how to construct a computer vision pipeline starting with foundational model and applying image augmentations. Most importantly, I learned how to use new tools such as **Vast.ai**'s cloud GPU computing.

### Future work 
There's so many different things to try out but with so little time because the bottleneck is dataloading. Each epoch with 16 workers takes 30 seconds. Some things to try out include different foundational models, different embedding dimensions, different loss functions + adjusting parameters for the loss functions, different ways to handle class imbalance, different number of epochs and learning rate, etc.

[Link to competition](https://www.kaggle.com/competitions/jaguar-re-id/)

---

## Anomaly Detection for Financial Data

Binary classification competition for anomaly detection in VDI process financial data. Dataset contained binary-encoded features from financial transactions requiring classification of anomalous vs. normal samples.

Systematically evaluated three modeling approaches of decreasing complexity:
1. PCA dimensionality reduction + Random Forest (F1: 22%)
2. Random Forest with hyperparameter tuning (F1: 57%)
3. Logistic Regression with default parameters (F1: 100%)

The perfect F1-score achieved with the simplest model demonstrated an important ML principle: always establish baseline performance with simple models before adding complexity. The linearly separable nature of the binary-encoded data made logistic regression the optimal choice, outperforming more sophisticated ensemble methods.

This project reinforced the value of systematic model selection and the importance of matching model complexity to problem complexity.

[Link to competition](https://www.kaggle.com/competitions/anomaly-dectection-for-financial)
