"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    rng = np.random.RandomState(seed)
    n, d = X.shape
    shuffled_indices = rng.permutation(n)
    X_shuffled, y_shuffled = X[shuffled_indices], y[shuffled_indices]
    return X_shuffled, y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n, d = X.shape
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    X_train, X_val, X_test = X[ : n_train], X[n_train : n_train + n_val], X[n_train + n_val : ] 
    y_train, y_val, y_test = y[ : n_train], y[n_train : n_train + n_val], y[n_train + n_val : ]
    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    std = np.std(X, axis=0)
    std[std==0] = 1.0
    return np.mean(X, axis=0), std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n, d = X.shape
    bias = np.ones((n , 1))
    return np.hstack([bias, X])

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X_new = standardize_features(X, mean, std)
    return add_bias_column(X_new)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    residuals = y_pred - y_true
    return np.mean(residuals ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n, _ = X.shape
    residual = y_pred - y_true
    return (2 / n) * X.T @ residual

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    A = X.T @ X
    b = X.T @ y
    return np.linalg.lstsq(X, y, rcond=None)[0]

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed != None:
        np.random.seed(seed)
        weights = np.random.normal(0, 0.01, n_features)
    else:
        weights = np.random.normal(0, 0.01, n_features)
    
    return weights

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    preds = predict_linear(X, weights)
    mse_grad = mse_gradient(X, y, preds)
    new_weights = weights - lr * mse_grad
    return new_weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    train_preds, val_preds = predict_linear(X_train, weights), predict_linear(X_val, weights)
    train_loss, val_loss = float(mse_loss(y_train, train_preds)), float(mse_loss(y_val, val_preds))
    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    stop = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_weights = weights
    else:
        wait += 1
        stop = (wait >= patience)
    
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    weights = initialize_weights(n_features, seed=seed)
    return {
        'weights' : weights,
        'best_weights' : weights.copy(),
        'best_val_loss' : np.inf,
        'wait' : 0,
        'train_losses' : [],
        'val_losses' : [],
        'stopped' : False
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    weights = state['weights']
    new_weights = gd_step(X_train, y_train, weights, lr)
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, new_weights)
    best_val_loss, wait, best_weights, stopped = update_early_stop_state(val_loss, state['best_val_loss'], state['wait'], new_weights, state['best_weights'], patience)

    state['weights'] = new_weights
    state['best_weights'] = best_weights
    state['best_val_loss'] = best_val_loss
    state['wait'] = wait
    state['stopped'] = stopped
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    _, n_features = X_train.shape
    state = init_training_state(n_features, seed=seed)
    for _ in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state['stopped'] is True:
            break
    
    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    residual = y_true - y_pred
    return np.mean(np.abs(residual))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    residual = y_pred - y_true
    return np.sqrt(np.mean(residual ** 2))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    residual = y_pred - y_true
    around_mean = y_true - np.mean(y_true)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(around_mean ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    return {
        'mae' : mae,
        'rmse' : rmse,
        'r2' : r2
    }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    # TODO: Return epoch indices and loss series for external plotting...
    epochs = list(range(1, len(train_losses) + 1))
    return epochs, [float(x) for x in train_losses], [float(x) for x in val_losses]

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    dist = w_gd - w_closed
    return np.linalg.norm(dist)

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # TODO: Build the initial LinearRegressionGD-style model dictionary...
    return {
        'learning_rate' : learning_rate,
        'epochs' : epochs,
        'patience' : patience,
        'seed' : seed,
        'weights' : None,
        'normal_weights' : None,
        'mean' : None,
        'std' : None,
        'train_losses' : [],
        'val_losses' : []
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    # TODO: Fit model with train stats, design matrices, GD, and normal eq
    learning_rate, epochs, patience, seed = model['learning_rate'], model['epochs'], model['patience'], model['seed']
    mean, std = compute_feature_stats(X_train)
    X_train_design, X_val_design = prepare_design_matrix(X_train, mean, std), prepare_design_matrix(X_val, mean, std)
    weights, train_losses, val_losses = train_batch_gd(X_train_design, y_train, X_val_design, y_val, learning_rate, epochs, patience, seed=seed)
    normal_weights = normal_equation(X_train_design, y_train)
    model['mean'] = mean
    model['std'] = std
    model['weights'] = weights
    model['normal_weights'] = normal_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses
    return model

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

