"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.asarray(X, dtype=float).copy()

    col_means = np.nanmean(X, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)

    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_means[cols]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    
    X = np.array(X, dtype=float)
    
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1
    
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    ratio_feat = numerator / (denominator + eps)

    return ratio_feat

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.array(X, dtype=float)
    col = np.array(col, dtype=float)
    
    return np.column_stack((X, col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.array(labels)

    classes, encoded = np.unique(labels, return_inverse=True)

    one_hot = np.eye(len(classes))[encoded]

    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.array(X, dtype=float)
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0.0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X = np.array(X, dtype=float)
    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)

    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    X = np.array(X, dtype=float)

    ones = np.ones(X.shape[0])
    
    return np.column_stack((ones, X))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.RandomState(seed)
    
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    indices = np.array(indices, dtype=int)

    X_subset = X[indices]
    y_subset = y[indices]

    return X_subset, y_subset

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    theta = np.linalg.pinv(X) @ y

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    X = np.array(X, dtype=float)
    theta = np.array(theta, dtype=float)

    y_pred = X @ theta

    return y_pred

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = np.abs(y_pred - y_true)
    return np.mean(mae)

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - y_true.mean()) ** 2)

    if SS_tot == 0:
        return 0.0

    return 1 - (SS_res/SS_tot)

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r = y_true - y_pred

    mean = np.mean(r)
    std = np.std(r)
    median_abs = np.median(np.abs(r))

    return {
        "mean": mean,
        "std": std,
        "median_abs": median_abs
    }

# Step 19 - prepare_cleaned_features
import numpy as np

def prepare_cleaned_features(X, iqr_k=1.5):
    X = np.array(X, dtype=float).copy()

    # Step 1: Impute missing values
    X = impute_nan_with_mean(X)

    # Step 2: Compute IQR bounds
    lower, upper = compute_iqr_bounds(X, k=iqr_k)

    # Step 3: Clip outliers
    X = clip_columns(X, lower, upper)

    return X

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    X_num = np.array(X_num, dtype=float)
    
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx
    ]
    ratio = make_ratio_feature(numerator, denominator, eps=1e-08)

    X_res = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_features = one_hot_encode(cat_labels)
        X_res = np.column_stack((X_res, cat_features))

    return X_res

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n_samples = X.shape[0]
    indices = make_shuffled_indices(n_samples, seed)

    
    train_indices, val_indices, test_indices = partition_indices(
        indices, train_ratio, val_ratio
    )

    X_train, y_train = subset_xy(X, y, train_indices)
    X_val, y_val = subset_xy(X, y, val_indices)
    X_test, y_test = subset_xy(X, y, test_indices)

    return {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test
}

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']

    mean, std = fit_standardizer(X_train)

    X_train = apply_standardizer(X_train, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_test = apply_standardizer(X_test, mean, std)

    X_train = add_bias_column(X_train)
    X_val = add_bias_column(X_val)
    X_test = add_bias_column(X_test)

    std_splits = {
        'X_train': X_train,
        'y_train': splits['y_train'],
        'X_val': X_val,
        'y_val': splits['y_val'],
        'X_test': X_test,
        'y_test': splits['y_test']
    }

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    return {
        "mae": mean_absolute_error(y_true, y_pred).astype(np.float64),
        "rmse": root_mean_squared_error(y_true, y_pred).astype(np.float64),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": {"mean": np.mean(y_true - y_pred), "std": np.std(y_true - y_pred), "median_abs": np.median(np.abs(y_true - y_pred))}
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(
    X, y, ratio_num_idx, ratio_den_idx,
    cat_labels=None, train_ratio=0.7,
    val_ratio=0.15, seed=42, iqr_k=1.5
):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # 1. Clean features
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)

    # 2. Feature engineering
    X_feat = assemble_feature_matrix(
        X_clean,
        ratio_num_idx,
        ratio_den_idx,
        cat_labels=cat_labels
    )

    # 3. Split dataset
    splits = make_train_val_test(
        X_feat, y, train_ratio, val_ratio, seed
    )

    # 4. Standardize and add bias
    std_splits, mean, std = standardize_and_add_bias(splits)

    # 5. Fit OLS
    theta = ols_fit(
        std_splits["X_train"],
        std_splits["y_train"]
    )

    # 6. Predict
    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)

    # 7. Evaluate
    val_metrics = evaluate_predictions(
        std_splits["y_val"], y_val_pred
    )

    test_metrics = evaluate_predictions(
        std_splits["y_test"], y_test_pred
    )

    # 8. Return results
    return {
        "theta": theta,
        "mean": mean,
        "std": std,
        "splits": std_splits,
        "val_predictions": y_val_pred,
        "test_predictions": y_test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }

