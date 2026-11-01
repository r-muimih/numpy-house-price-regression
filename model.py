"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()

    nan_mask = np.isnan(out)

    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(out, axis=0)

    # Columns that are entirely NaN -> nanmean gives NaN; replace with 0.0
    all_nan_cols = np.isnan(col_means)
    col_means = np.where(all_nan_cols, 0.0, col_means)

    # Broadcast column means into the NaN positions
    row_idx, col_idx = np.where(nan_mask)
    out[row_idx, col_idx] = col_means[col_idx]

    return out

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    """Compute per-column outlier clip bounds using the IQR rule.

    Args:
        X: (N, F) array-like of floats.
        k: multiplier applied to the IQR to widen the bounds (default 1.5).

    Returns:
        (lower, upper): two 1-D float ndarrays of shape (F,), where
            lower = q1 - k * iqr
            upper = q3 + k * iqr
    """
    X = np.asarray(X, dtype=np.float64)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
import numpy as np

def clip_columns(X, lower, upper):
    """Clip every entry of X to per-column [lower, upper] bounds.

    Args:
        X: (N, F) array-like of floats.
        lower: 1-D array-like of length F, per-column lower bounds.
        upper: 1-D array-like of length F, per-column upper bounds.

    Returns:
        (N, F) float ndarray, same shape as X, with each column clipped
        independently to [lower[j], upper[j]]. X is not modified.
    """
    X = np.asarray(X, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
import numpy as np

def make_ratio_feature(numerator, denominator, eps=1e-8):
    """Form a derived ratio feature via safe division.

    Args:
        numerator: (N,) array-like of floats.
        denominator: (N,) array-like of floats.
        eps: small constant added to the denominator to avoid division by zero.

    Returns:
        (N,) float ndarray equal to numerator / (denominator + eps).
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    """Horizontally append a 1-D feature column onto a 2-D design matrix.

    Args:
        X: (N, F) array-like of floats.
        col: (N,) array-like of floats to append as a new last column.

    Returns:
        (N, F+1) float ndarray equal to X with col appended as the last column.
        X is not modified.
    """
    X = np.asarray(X, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64).reshape(-1, 1)

    return np.hstack([X, col])

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    """Convert a 1-D array of categorical labels into a dense one-hot matrix.

    Column j corresponds to the j-th unique label in sorted order.

    Args:
        labels: (N,) array-like of categorical labels (any hashable/sortable type).

    Returns:
        (N, C) float ndarray, where C is the number of unique labels; each row
        has a single 1.0 in the column matching that sample's label, 0.0 elsewhere.
    """
    labels = np.asarray(labels)

    categories, inverse = np.unique(labels, return_inverse=True)
    N = labels.shape[0]
    C = categories.shape[0]

    one_hot = np.zeros((N, C), dtype=np.float64)
    one_hot[np.arange(N), inverse] = 1.0

    return one_hot

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    """Compute per-column mean and std used to standardize features.

    Args:
        X: (N, F) array-like of floats.

    Returns:
        (mean, std): two 1-D float ndarrays of shape (F,). std entries that
        would be exactly 0 are replaced with 1.0 so later division is safe.
    """
    X = np.asarray(X, dtype=np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0.0, 1.0, std)

    return mean, std

# Step 8 - apply_standardizer
import numpy as np

def apply_standardizer(X, mean, std):
    """Standardize X using previously fitted per-column mean/std.

    Args:
        X: (N, F) array-like of floats.
        mean: (F,) array-like, per-column mean.
        std: (F,) array-like, per-column standard deviation (no zeros expected).

    Returns:
        (N, F) float ndarray equal to (X - mean) / std, broadcast per column.
        Inputs are not modified.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    return (X - mean) / std

# Step 9 - add_bias_column
import numpy as np

def add_bias_column(X):
    """Prepend a column of ones to X so OLS can learn an intercept term.

    Args:
        X: (N, F) array-like of floats.

    Returns:
        (N, F+1) float ndarray whose column 0 is all 1.0 and whose remaining
        columns are identical to X. X is not modified.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)

    return np.hstack([ones, X])

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    """Create a reproducibly shuffled permutation of row indices.

    Args:
        n_samples: number of samples (indices 0..n_samples-1).
        seed: integer seed controlling the shuffle order.

    Returns:
        1-D int ndarray of length n_samples, a permutation of
        0..n_samples-1. The same (n_samples, seed) always yields the
        same order.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
import numpy as np

def partition_indices(indices, train_ratio, val_ratio):
    """Split a 1-D index array into contiguous train/val/test slices.

    Args:
        indices: 1-D array-like of row indices (e.g. a shuffled permutation).
        train_ratio: fraction of N to allocate to train, in (0, 1).
        val_ratio: fraction of N to allocate to validation, in (0, 1),
            with train_ratio + val_ratio < 1.

    Returns:
        (train_idx, val_idx, test_idx): three 1-D int ndarrays that together
        form a partition of `indices`, taken as contiguous prefixes:
        train gets floor(N * train_ratio) entries, val gets the next
        floor(N * val_ratio) entries, and test gets the remainder.
    """
    indices = np.asarray(indices)
    N = indices.shape[0]

    n_train = int(np.floor(N * train_ratio))
    n_val = int(np.floor(N * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
import numpy as np

def subset_xy(X, y, indices):
    """Select rows of X and y at the given indices, preserving row order.

    Args:
        X: (N, F) array-like of floats.
        y: (N,) array-like of floats (target values).
        indices: 1-D array-like of integer row indices to select.

    Returns:
        (X_sub, y_sub): X_sub has shape (len(indices), F), y_sub has shape
        (len(indices),), both ordered to match `indices`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    indices = np.asarray(indices)

    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub

# Step 13 - ols_fit
import numpy as np

def ols_fit(X, y):
    """Solve for OLS weights minimizing ||X theta - y||_2^2.

    Args:
        X: (N, D) array-like of floats, design matrix (bias column already included).
        y: (N,) array-like of floats, target values.

    Returns:
        theta: (D,) float ndarray of fitted weights.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    theta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    return theta

# Step 14 - ols_predict
import numpy as np

def ols_predict(X, theta):
    """Score new rows with a fitted linear model: predictions = X @ theta.

    Args:
        X: (N, D) array-like of floats, design matrix (same column layout as
            the one theta was fit on, including any bias column).
        theta: (D,) array-like of floats, fitted weight vector.

    Returns:
        (N,) float ndarray of predicted targets.
    """
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    return X @ theta

# Step 15 - mean_absolute_error
import numpy as np

def mean_absolute_error(y_true, y_pred):
    """Compute the mean absolute error between true and predicted targets.

    Args:
        y_true: (N,) array-like of floats, ground-truth targets.
        y_pred: (N,) array-like of floats, predicted targets.

    Returns:
        Python float, the average of |y_true - y_pred| over all elements.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean(np.abs(y_true - y_pred)))

# Step 16 - root_mean_squared_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    """Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares around the mean of y_true.

    Args:
        y_true: (N,) array-like of floats, ground-truth targets.
        y_pred: (N,) array-like of floats, predicted targets.

    Returns:
        Python float. Returns 0.0 if SS_tot is exactly 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred):
    """Summarize prediction residuals r = y_true - y_pred.

    Args:
        y_true: (N,) array-like of floats, ground-truth targets.
        y_pred: (N,) array-like of floats, predicted targets.

    Returns:
        dict with keys:
            'mean': Python float, mean of residuals.
            'std': Python float, population std (ddof=0) of residuals.
            'median_abs': Python float, median of |residuals|.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    r = y_true - y_pred

    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "median_abs": float(np.median(np.abs(r))),
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X_imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imputed, k=iqr_k)
    X_clean = clip_columns(X_imputed, lower, upper)

    return X_clean

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    """Build an extended feature matrix from numeric columns plus a derived ratio
    and an optional one-hot categorical block.

    Args:
        X_num: (N, F) array-like of floats, cleaned numeric feature matrix.
        ratio_num_idx: int column index into X_num used as the ratio numerator.
        ratio_den_idx: int column index into X_num used as the ratio denominator.
        cat_labels: optional (N,) array-like of categorical labels to one-hot
            encode and append. If None, no categorical block is added.

    Returns:
        (N, F+1) float ndarray, or (N, F+1+C) if cat_labels is provided,
        where C is the number of unique categories.
    """
    X_num = np.asarray(X_num, dtype=np.float64)

    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(numerator, denominator)

    X_out = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)
        X_out = np.hstack([X_out, cat_block])

    return X_out

# Step 21 - make_train_val_test
import numpy as np

def make_shuffled_indices(n_samples, seed):
    """Create a reproducibly shuffled permutation of row indices.

    Args:
        n_samples: number of samples (indices 0..n_samples-1).
        seed: integer seed controlling the shuffle order.

    Returns:
        1-D int ndarray of length n_samples, a permutation of
        0..n_samples-1. The same (n_samples, seed) always yields the
        same order.
    """
    rng = np.random.RandomState(seed)
    return rng.permutation(n_samples)


def partition_indices(indices, train_ratio, val_ratio):
    """Split a 1-D index array into contiguous train/val/test slices."""
    indices = np.asarray(indices)
    N = indices.shape[0]

    n_train = int(np.floor(N * train_ratio))
    n_val = int(np.floor(N * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def subset_xy(X, y, indices):
    """Select rows of X and y at the given indices, preserving row order."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    indices = np.asarray(indices)

    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub


def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    """Shuffle and materialize train/val/test splits from X and y.

    Args:
        X: (N, F) array-like of floats, feature matrix.
        y: (N,) array-like of floats, target vector.
        train_ratio: fraction of N allocated to train, in (0, 1).
        val_ratio: fraction of N allocated to validation, in (0, 1),
            with train_ratio + val_ratio < 1.
        seed: integer seed controlling the reproducible shuffle order.

    Returns:
        dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test',
        'y_test', where each X/y pair shares matching row order.
    """
    N = X.shape[0] if hasattr(X, "shape") else len(X)

    shuffled_idx = make_shuffled_indices(N, seed)
    train_idx, val_idx, test_idx = partition_indices(shuffled_idx, train_ratio, val_ratio)

    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    """Fit standardization on train features, apply to all splits, add bias.

    Args:
        splits: dict with keys 'X_train', 'y_train', 'X_val', 'y_val',
            'X_test', 'y_test'.

    Returns:
        (std_splits, mean, std):
            std_splits: dict with the same keys as `splits`. Each X_* has
                been standardized using train-fit statistics and had a bias
                column prepended; each y_* is left unchanged.
            mean: (F,) float ndarray, per-feature mean fit on X_train.
            std: (F,) float ndarray, per-feature std fit on X_train
                (zeros replaced with 1.0).
    """
    mean, std = fit_standardizer(splits["X_train"])

    std_splits = {}
    for name in ("X_train", "X_val", "X_test"):
        X_scaled = apply_standardizer(splits[name], mean, std)
        std_splits[name] = add_bias_column(X_scaled)

    for name in ("y_train", "y_val", "y_test"):
        std_splits[name] = splits[name]

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    """Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.

    Args:
        y_true: (N,) array-like of floats, ground-truth targets.
        y_pred: (N,) array-like of floats, predicted targets.

    Returns:
        dict with keys:
            'mae': Python float, mean absolute error.
            'rmse': Python float, root mean squared error.
            'r2': Python float, coefficient of determination.
            'residual_summary': dict with keys 'mean', 'std', 'median_abs'.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred),
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(
    X,
    y,
    ratio_num_idx,
    ratio_den_idx,
    cat_labels=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    iqr_k=1.5,
):
    """Run the full clean -> featurize -> split -> standardize -> OLS -> evaluate pipeline.

    Args:
        X: (N, F) array-like of raw numeric features, may contain NaN/outliers.
        y: (N,) array-like of target house prices.
        ratio_num_idx: int column index into cleaned X used as ratio numerator.
        ratio_den_idx: int column index into cleaned X used as ratio denominator.
        cat_labels: optional (N,) array-like of categorical labels to one-hot
            encode and append.
        train_ratio: fraction of N allocated to train.
        val_ratio: fraction of N allocated to validation.
        seed: integer seed controlling the reproducible shuffle order.
        iqr_k: IQR multiplier for outlier clipping (default 1.5).

    Returns:
        dict with keys:
            'theta': (D,) float ndarray, fitted OLS weights.
            'y_test': (N_test,) float ndarray, held-out test targets.
            'y_test_pred': (N_test,) float ndarray, held-out test predictions.
            'test_metrics': dict from evaluate_predictions on the test split.
            'val_metrics': dict from evaluate_predictions on the val split.
    """
    # 1. Clean raw numeric features
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)

    # 2. Feature assembly: derived ratio + optional categorical one-hots
    X_features = assemble_feature_matrix(
        X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels
    )

    # 3. Reproducible train/val/test split
    splits = make_train_val_test(X_features, y, train_ratio, val_ratio, seed)

    # 4. Train-only standardization + bias column
    std_splits, mean, std = standardize_and_add_bias(splits)

    # 5. OLS fit via the normal equation (on train only)
    theta = ols_fit(std_splits["X_train"], std_splits["y_train"])

    # 6. Predict + evaluate on val and test
    y_val_pred = ols_predict(std_splits["X_val"], theta)
    val_metrics = evaluate_predictions(std_splits["y_val"], y_val_pred)

    y_test_pred = ols_predict(std_splits["X_test"], theta)
    test_metrics = evaluate_predictions(std_splits["y_test"], y_test_pred)

    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
    }

