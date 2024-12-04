# Groundwater Level Prediction - Hi!ckathon #5

This repository contains the solution developed for **Hi!ckathon #5**, a competitive AI event organized by **Hi! Paris** in collaboration with **École Polytechnique** and **HEC Paris**. The challenge revolved around utilizing AI to predict groundwater levels during summer months in France, addressing sustainability concerns through cutting-edge machine learning.

## Challenge Overview

The dataset provided was extensive, featuring:
- **3 million+ data points** in the training dataset
- Over **100 features** including weather, hydrology, abstraction, socio-economic data, and piezometry
- Numerous missing values across both numerical and categorical features

The primary objective was to build a robust machine learning model capable of predicting groundwater level categories during the summer, balancing accuracy, interpretability, and computational efficiency.

---

## Approach and Methodology

### 1. Data Preprocessing
- **Handling Missing Values:**
  - *KNN Imputer* for categorical features.
  - *Iterative Imputer* for numerical features.
  - Features with over **80% missing values** were excluded.
- **Feature Engineering:**
  - Label encoding for categorical variables.
  - Standard scaling for numerical variables.
- **Feature Selection:**
  - Utilized the feature importance method from *RandomForestClassifier* to filter features based on a threshold of **0.01** importance.

### 2. Model Development
- **Baseline Model:**
  - Trained a *RandomForestClassifier* on the preprocessed dataset.
  - Utilized feature importance to select the most relevant features and retrained the model.
- **Advanced Models:**
  - Experimented with:
    - *Neural Networks* using ReLU activation, Batch Normalization, and Dropout regularization.
    - Boosting algorithms (*XGBoost*, *CatBoost*, *HistGradientBoost*).
  - Achieved comparable performance with *CatBoost* and *HistGradientBoost*.

### 3. Performance Evaluation
- The **RandomForestClassifier with selected features** outperformed other models in terms of accuracy and computational efficiency.
- Due to time constraints, advanced hyperparameter tuning techniques like *Grid Search CV* were not implemented.

---

## Key Results
- **Best Performing Model:** RandomForestClassifier with selected features.
- **Limitations:**
  - Overfitting observed in initial models.
  - High computational cost for imputers due to the dataset's size.

---

## Future Improvements
1. **Optimizing Missing Data Handling:**
   - Explore scalable methods like matrix factorization or deep learning-based imputers.
2. **Feature Engineering:**
   - Use advanced techniques such as polynomial features, interaction terms, and domain-specific transformations.
3. **Hyperparameter Optimization:**
   - Apply *Grid Search CV* or *Random Search CV* with parallel processing for tuning.
4. **Overfitting Mitigation:**
   - Incorporate cross-validation, ensemble averaging, and constrained tree models.
5. **Ensemble Models:**
   - Implement stacking or blending to combine model strengths.
6. **Exploration of Time-Series Analysis:**
   - Frame the task as a time-series problem to enhance predictive power.

---

## Deliverables
- **Data Science Model:** Code implementing the model and preprocessing steps (`.ipynb`/`.py`).
- **Predictions:** Test set predictions submitted to the leaderboard (`.csv`).
- **Scientific and Business Report:** PDF outlining the approach, results, and societal impact.
- **Presentation Video:** 3-minute video pitch summarizing the project.

---

## Acknowledgements
This project was developed during **Hi!ckathon #5**, hosted by Hi! Paris, École Polytechnique, and HEC Paris. We extend our gratitude to the event organizers and mentors for their guidance.

---

