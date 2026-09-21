# ??? NYC Airbnb Room Type Predictor

A machine learning classification project that predicts the **room type** of a New York City Airbnb listing — *Entire home/apt*, *Private room*, or *Shared room* — based on listing attributes such as price, location, neighbourhood, and availability.

---

## ?? Project Overview

The NYC Airbnb Open Data (2019) contains ~48,900 listings across five boroughs of New York City. This project builds an end-to-end ML pipeline that:

1. **Explores** the data with rich visualizations  
2. **Cleans & preprocesses** features (imputation, scaling, encoding)  
3. **Trains & compares** multiple classifiers  
4. **Tunes** the best model with `RandomizedSearchCV`  
5. **Evaluates** final performance on a held-out test set  

---

## ?? Dataset

| Property | Value |
|----------|-------|
| **Source** | [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) |
| **File** | `AB_NYC_2019.csv` |
| **Rows** | 48,895 listings |
| **Columns** | 16 features |
| **Target** | `room_type` (3 classes) |

### Target Classes

| Class | Description |
|-------|-------------|
| `Entire home/apt` | Full apartment/home exclusively rented |
| `Private room` | A private room within a shared home |
| `Shared room` | A shared room (e.g., hostel-style) |

---

## ??? Project Structure

```
NYC_Airbnb_Room_Type_Predictor/
¦
+-- NYC_Airbnb_Room_Type_Predictor.ipynb   # Main Jupyter Notebook
+-- README.md                              # Project documentation
```

---

## ?? Exploratory Data Analysis (EDA)

The notebook performs in-depth EDA including:

- **Class distribution** — `room_type` count plot reveals class imbalance (Entire home/apt dominates)
- **Numerical distributions** — Histograms of `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`
- **Borough breakdown** — Count plot by `neighbourhood_group` (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
- **Price vs. Room Type** — Box plot showing price distribution per room type
- **Correlation Heatmap** — Feature correlations among numerical variables
- **Geo-scatter Plot** — Latitude/longitude scatter coloured by room type to reveal spatial clustering

---

## ?? Feature Engineering & Preprocessing

### Dropped Columns
Identifiers and temporal columns not useful for prediction are dropped:

```
id, name, host_id, host_name, last_review
```

### Missing Values
- `reviews_per_month` — NaN values filled with `0` (no reviews yet ? 0 reviews/month)

### Outlier Capping
- `price` — clipped at the **99th percentile**
- `minimum_nights` — clipped at the **99th percentile**

### Sklearn Preprocessing Pipeline

| Feature Type | Columns | Steps |
|---|---|---|
| **Numerical** | `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365` | Median Imputation ? StandardScaler |
| **Categorical** | `neighbourhood_group`, `neighbourhood` | Most-Frequent Imputation ? OneHotEncoder |

---

## ?? Models & Results

### Train/Test Split
- **Test size**: 33%  
- **Stratified** split to preserve class proportions

### Baseline Model Comparison (3-fold Cross-Validation on Training Set)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Decision Tree | 0.782 | 0.647 |
| Gradient Boosting | 0.850 | 0.705 |
| **Random Forest** | **0.851** | **0.715** |

> Macro F1 is used as the primary metric since it treats all classes equally — important given the class imbalance.

### Hyperparameter Tuning — Random Forest (RandomizedSearchCV)

```python
param_distribution = {
    "classifier__n_estimators"     : [100, 150, 200, 300],
    "classifier__max_depth"        : [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
# 10 iterations, 3-fold CV, scoring = f1_macro
```

**Best Parameters Found:**
```
n_estimators      = 200
min_samples_split = 10
max_depth         = None
```

**Best CV Macro-F1:** `0.7307`

### Final Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **85.66%** |
| **Macro F1** | **0.7402** |

---

## ??? Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML pipeline, models, evaluation |
| `kagglehub` | Dataset download from Kaggle |

---

## ?? Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/salman1451/NYC_Airbnb_Room_Type_Predictor.git
   cd NYC_Airbnb_Room_Type_Predictor
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook NYC_Airbnb_Room_Type_Predictor.ipynb
   ```

3. The notebook uses `kagglehub` to automatically download the dataset. Make sure you have your Kaggle API credentials configured, or the notebook is run on Google Colab (which uses a cached version).

---

## ?? Key Insights

- **Location matters**: Latitude/longitude features are strong predictors of room type, since entire apartments cluster in Manhattan while private rooms are more distributed across outer boroughs.
- **Price** is a highly informative feature — entire homes command significantly higher prices than private or shared rooms.
- **Class imbalance** (`Shared room` is a small minority class) is handled via `class_weight="balanced"` in tree models and by evaluating with Macro F1.
- **Random Forest** outperforms Decision Tree and Gradient Boosting across both accuracy and F1, and is the selected final model.

---

## ?? License

This project is open-source and available under the [MIT License](LICENSE).

---

## ?? Acknowledgements

- Dataset: [Dgomonov – New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) on Kaggle
- Built with ?? using Python and scikit-learn
