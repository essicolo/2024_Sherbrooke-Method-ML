import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sherbrooke Method Modelling

    ## Principles

    **Calibration of the Probe**. Starting from particle-size parameters $g_1$ and $g_2$, as well as the specific gravity $Gs$, we predict an *exponential association* curve comprising three parameters, $y_{min}$, $y_{max}$, and the $slope$ which best predicts the soil water content $θ$ as a function of the probe measurement $M$, based on Proctor tests.

    **Sherbrooke Method**. The particle-size parameters $g_1$ and $g_2$, the specific gravity $Gs$, as well as the water content measured before flooding $θ_{R1}$, allow estimating the degree of saturation obtained after one minute of wetting $S_{opt}$. With $S_{opt}$ and the water content measured after the flood $θ_{R2}$, we can obtain the dry density $ρ_{d}$ with $ρ_w$ and the $Gs$.

    An AI agent was used at the very end of the analysis to polish the code, include safeguards and to implement the config dictionnary. 

    ## Packages

    We will need the Polars module for importing and manipulating tabular data, Numpy for matrix calculations, Scikit-learn for machine learning and, for graphics, Matplotlib and Let's-plot.
    """
    )
    return


@app.cell
def _(config):
    # notebook
    import marimo as mo
    from tqdm import tqdm

    # math
    import polars as pl
    import pandas as pd
    import numpy as np
    from scipy.optimize import minimize
    import functools

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    from great_tables import GT

    # model
    import optuna
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import cross_val_score, GroupKFold, train_test_split
    from sklearn.metrics import (
        mean_squared_error,
        root_mean_squared_error,
        r2_score, make_scorer
    )
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.base import BaseEstimator, TransformerMixin

    np.random.seed(config["MASTER_SEED"])
    return (
        BaseEstimator,
        GT,
        GaussianProcessRegressor,
        GroupKFold,
        Matern,
        RobustScaler,
        TransformerMixin,
        WhiteKernel,
        cross_val_score,
        make_scorer,
        mean_squared_error,
        mo,
        np,
        optuna,
        pd,
        pl,
        plt,
        root_mean_squared_error,
        sns,
        tqdm,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Configuration and Setup

    This section establishes the computational environment with reproducible random seeds,
    optimization parameters, and file paths for the analysis pipeline.
    """
    )
    return


@app.cell
def _():
    config = {
        # Reproducibility, numbers from random.org, between 0 and 1000000
        "MASTER_SEED": 592064,
        "PROBE_MODEL_SEED": 908493,
        "SR_MODEL_SEED": 71364,
        "BOOTSTRAP_SEED": 381058,
        "LOOCV_SEED": 145018,
        "EXAMPLE_SEED": 643126,

        # File paths
        "DATA_PROCTOR": "data/r_proctor.csv",
        "DATA_SOILS": "data/soils_rosin.csv",
        "DATA_FIELD": "data/r_field.csv",
        "OUTPUT_PRED_STATS": "data/pred_stats_summary.csv",
        "OUTPUT_ACCURACY": "data/accuracy_summary.csv",

        # GP Optimization parameters
        "GP_MAX_ITER_LBFGS": 50000,
        "GP_WHITE_KERNEL_NOISE": 0.01,
        "GP_WHITE_KERNEL_NOISE_BOUNDS": (1e-3, 1.0),  
        "GP_NORMALIZE_Y": True,
        "GP_COPY_X_TRAIN": False,
        "GP_ALPHA_BOUNDS": (0.01, 1.0), 
        "GP_NU_BOUNDS": (0.5, 2.5), 
        "GP_LENGTH_SCALE_BOUNDS": (0.01, 10.0),
        "GP_LENGTH_SCALE_SEARCH_BOUNDS": (0.1, 5.0),
        "GP_N_RESTARTS_OPTIMIZER": 5,

        # Optimization parameters
        "OPTUNA_N_TRIALS": 30,
        "CV_N_SPLITS": 5, # just for optuna, the model is validated on leave one-soil out, which is more reliable
        "N_RESTARTS_OPTIMIZER": 5,
        "BOOTSTRAP_N_SAMPLES": 500,

        # Convergence and overfitting parameters
        "OVERFITTING_VAL_SIZE": 0.2,
        "OVERFITTING_RATIO_THRESHOLD": 1.5,
        "OVERFITTING_PENALTY_WEIGHT": 0.5,
        "CV_VARIANCE_WEIGHT": 0.3,
        "GP_N_RESTARTS_TEMP": 2, 

        # Physical constants
        "WATER_DENSITY": 1000.0,  # kg/m³
        "MAX_SATURATION": 0.99,

        # Plotting parameters
        "PLOT_N_ROWS": 8,
        "PLOT_N_COLS": 4,
        "PROBE_RANGE_MIN": 1800,
        "PROBE_RANGE_MAX": 2800,
        "N_PROBE_POINTS": 100,
    }
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Utility Functions

    Mathematical transformations and helper functions for bounded variable handling,
    phase relationships, and model evaluation metrics.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    GaussianProcessRegressor,
    GroupKFold,
    Matern,
    RobustScaler,
    WhiteKernel,
    config,
    cross_val_score,
    make_scorer,
    mean_squared_error,
    np,
    optuna,
    train_test_split,
):
    def volumetric_water_content_to_log_ratio(vwc):
        """
        Transform volumetric water content to water log ratio for unbounded modeling.

        Args:
            vwc: Volumetric water content [0, 1]

        Returns:
            Water log ratio [-∞, +∞]
        """
        return np.log(vwc / (1 - vwc))

    def log_ratio_to_volumetric_water_content(wlr):
        """
        Transform water log ratio back to volumetric water content.

        Args:
            wlr: Water log ratio [-∞, +∞]

        Returns:
            Volumetric water content [0, 1]
        """
        return np.exp(wlr) / (1 + np.exp(wlr))

    def saturation_ratio_to_log_ratio(sr):
        """
        Transform saturation ratio to log ratio for unbounded modeling.

        Args:
            sr: Saturation ratio [0, 1]

        Returns:
            Saturation log ratio [-∞, +∞]
        """
        return np.log(sr / (1 - sr))

    def log_ratio_to_saturation_ratio(srlr, max_sr=None):
        """
        Transform saturation log ratio back to saturation ratio.

        Args:
            srlr: Saturation log ratio [-∞, +∞]
            max_sr: Maximum allowable saturation ratio

        Returns:
            Saturation ratio [0, max_sr or config["MAX_SATURATION"]]
        """
        if max_sr is None:
            max_sr = config["MAX_SATURATION"]
        return np.minimum(max_sr, np.exp(srlr) / (1 + np.exp(srlr)))


    def rmse_vwc(y_true_wlr, y_pred_wlr):
        y_true_vwc = log_ratio_to_volumetric_water_content(y_true_wlr)
        y_pred_vwc = log_ratio_to_volumetric_water_content(y_pred_wlr)
        return np.sqrt(mean_squared_error(y_true_vwc, y_pred_vwc))

    rmse_vwc_scorer = make_scorer(rmse_vwc, greater_is_better=False)

    def phase_ρd(theta, sr, rho_w, gs):
        """
        Compute dry density from phase relationships.

        Args:
            theta: Volumetric water content
            sr: Degree of saturation
            rho_w: Water density (kg/m³)
            gs: Specific gravity

        Returns:
            Dry density (kg/m³)
        """
        return (1 - theta / sr) * gs * rho_w

    def phase_Sr(theta, rho_d, rho_w, gs):
        """
        Compute saturation ratio from phase relationships.

        Args:
            theta: Volumetric water content
            rho_d: Dry density (kg/m³)
            rho_w: Water density (kg/m³)
            gs: Specific gravity

        Returns:
            Degree of saturation
        """
        return theta / (1 - rho_d / (gs * rho_w))

    def porosity(rho_d, rho_w, gs):
        """
        Compute porosity from dry density and specific gravity.

        Args:
            rho_d: Dry density (kg/m³)
            rho_w: Water density (kg/m³)
            gs: Specific gravity

        Returns:
            Porosity [0, 1]
        """
        return 1 - rho_d / (gs * rho_w)

    def mean_variance_index(scores, weight=0.7):
        """
        Combine mean and variance for optimization objectives.

        Args:
            scores: Array of cross-validation scores
            weight: Weight for mean vs. variance trade-off

        Returns:
            Combined score for maximization
        """
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        score_range = np.max(scores) - np.min(scores)

        if score_range == 0:
            return mean_score

        normalized_mean = (mean_score - np.min(scores)) / score_range
        normalized_std = 1 - (std_score / np.max(scores))

        return weight * normalized_mean + (1 - weight) * normalized_std

    def optimize_gaussian_process_hyperparameters(
        features, targets, config, study_name, random_seed, soil_ids=None
    ):
        """
        Enhanced Optuna optimization for Gaussian Process hyperparameters with overfitting control.

        Args:
            features: Preprocessed feature matrix
            targets: Target variable array
            config: Configuration object with optimization parameters
            study_name: Name for the Optuna study
            random_seed: Random seed for reproducibility
            soil_ids: Optional array of soil IDs for group-based CV

        Returns:
            Best hyperparameters dictionary with validation metrics
        """

        np.random.seed(random_seed)

        def objective(trial):
            alpha = trial.suggest_float("alpha", *config["GP_ALPHA_BOUNDS"], log=True)
            nu = trial.suggest_float("nu", *config["GP_NU_BOUNDS"])

            # Add length scale as a hyperparameter for better regularization control
            length_scale = trial.suggest_float(
                "length_scale", 
                *config.get("GP_LENGTH_SCALE_SEARCH_BOUNDS", (0.1, 10.0)), 
                log=True
            )

            kernel = Matern(
                length_scale=length_scale,
                nu=nu,
                length_scale_bounds=config["GP_LENGTH_SCALE_BOUNDS"]
            ) + WhiteKernel(
                noise_level=config.get("GP_WHITE_KERNEL_NOISE", 0.1),
                noise_level_bounds=config.get("GP_WHITE_KERNEL_NOISE_BOUNDS", (1e-8, 1.0))
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=config.get("GP_N_RESTARTS_OPTIMIZER", 3),
                copy_X_train=config.get("GP_COPY_X_TRAIN", False),
                normalize_y=config.get("GP_NORMALIZE_Y", True),
                optimizer='fmin_l_bfgs_b',
            )

            # Step-by-step: create scaler and scale features
            scaler_cv = RobustScaler()
            features_scaled_cv = scaler_cv.fit_transform(features)

            # Use GroupKFold if soil_ids are provided to prevent data leakage
            cv = GroupKFold(n_splits=min(config["CV_N_SPLITS"], len(np.unique(soil_ids))))
            cv_scores = cross_val_score(
                gp, features_scaled_cv, targets, groups=soil_ids, cv=cv, scoring="neg_root_mean_squared_error"
            )


            # Additional overfitting detection: train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, 
                test_size=config.get("OVERFITTING_VAL_SIZE", 0.2), 
                random_state=random_seed + trial.number
            )

            # Step-by-step: fit scaler on training data
            scaler_temp = RobustScaler()
            X_train_scaled_temp = scaler_temp.fit_transform(X_train)
            X_val_scaled_temp = scaler_temp.transform(X_val)

            # Create and fit GP model for validation
            gp_temp = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=alpha, 
                n_restarts_optimizer=config.get("GP_N_RESTARTS_TEMP", 3), 
                copy_X_train=False, 
                normalize_y=config.get("GP_NORMALIZE_Y", True)
            )
            gp_temp.fit(X_train_scaled_temp, y_train)

            # Make predictions on scaled data
            train_pred = gp_temp.predict(X_train_scaled_temp)
            val_pred = gp_temp.predict(X_val_scaled_temp)

            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            # Overfitting penalty: if validation RMSE >> training RMSE
            overfitting_ratio = val_rmse / (train_rmse + 1e-8)
            overfitting_threshold = config.get("OVERFITTING_RATIO_THRESHOLD", 1.5)
            penalty_weight = config.get("OVERFITTING_PENALTY_WEIGHT", 0.5)
            overfitting_penalty = max(0, overfitting_ratio - overfitting_threshold) * penalty_weight

            # Combine CV score with stability and overfitting metrics
            cv_mean = -np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Favor models with good CV performance, low variance, and low overfitting
            variance_weight = config.get("CV_VARIANCE_WEIGHT", 0.3)
            score = cv_mean - cv_std * variance_weight - overfitting_penalty

            # Store additional metrics for analysis
            trial.set_user_attr("cv_std", cv_std)
            trial.set_user_attr("train_rmse", train_rmse)
            trial.set_user_attr("val_rmse", val_rmse)
            trial.set_user_attr("overfitting_ratio", overfitting_ratio)

            return -score  # Optuna minimizes, we want to minimize RMSE

        study = optuna.create_study(study_name=study_name, direction="minimize")
        study.optimize(objective, n_trials=config["OPTUNA_N_TRIALS"], show_progress_bar=True)

        # Print overfitting analysis
        best_trial = study.best_trial
        print(f"\nOptimization Results for {study_name}:")
        print(f"  Best CV RMSE: {best_trial.value:.4f}")
        print(f"  CV Std: {best_trial.user_attrs.get('cv_std', 'N/A'):.4f}")
        print(f"  Training RMSE: {best_trial.user_attrs.get('train_rmse', 'N/A'):.4f}")
        print(f"  Validation RMSE: {best_trial.user_attrs.get('val_rmse', 'N/A'):.4f}")
        print(f"  Overfitting Ratio: {best_trial.user_attrs.get('overfitting_ratio', 'N/A'):.2f}")

        overfitting_ratio = best_trial.user_attrs.get('overfitting_ratio', 1.0)
        warning_threshold = config.get("OVERFITTING_WARNING_THRESHOLD", 2.0)
        caution_threshold = config.get("OVERFITTING_CAUTION_THRESHOLD", 1.5)

        if overfitting_ratio > warning_threshold:
            print(f"  ⚠️  WARNING: Potential overfitting detected (ratio: {overfitting_ratio:.2f})")
        elif overfitting_ratio > caution_threshold:
            print(f"  ⚠️  CAUTION: Moderate overfitting detected (ratio: {overfitting_ratio:.2f})")
        else:
            print(f"  ✓ Overfitting appears controlled (ratio: {overfitting_ratio:.2f})")

        return study.best_params

        return study.best_params

    def create_stratified_soil_splits(df, test_size=0.2, random_state=None):
        """
        Create stratified train/test splits by Soil_ID to prevent data leakage.

        Args:
            df: DataFrame with Soil_ID column
            test_size: Proportion of soils for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_soil_ids, test_soil_ids)
        """
        from sklearn.model_selection import train_test_split

        unique_soil_ids = df["Soil_ID"].unique().to_list()

        if len(unique_soil_ids) < 5:
            return unique_soil_ids, []

        train_soils, test_soils = train_test_split(
            unique_soil_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        return train_soils, test_soils

    def split_data_by_soil_ids(df, train_soil_ids, test_soil_ids):
        """
        Split DataFrame based on Soil_ID lists.

        Args:
            df: Input DataFrame
            train_soil_ids: List of soil IDs for training
            test_soil_ids: List of soil IDs for testing

        Returns:
            Tuple of (train_df, test_df)
        """
        import polars as pl
        train_df = df.filter(pl.col("Soil_ID").is_in(train_soil_ids))
        test_df = df.filter(pl.col("Soil_ID").is_in(test_soil_ids))

        return train_df, test_df
    return (
        create_stratified_soil_splits,
        log_ratio_to_saturation_ratio,
        log_ratio_to_volumetric_water_content,
        optimize_gaussian_process_hyperparameters,
        phase_Sr,
        phase_ρd,
        porosity,
        rmse_vwc_scorer,
        saturation_ratio_to_log_ratio,
        split_data_by_soil_ids,
        volumetric_water_content_to_log_ratio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    The `proctor` table includes data from the Proctor tests, `soil` includes the optimized particle-size parameters from the notebook `01_multilevel_rosin.ipynb`, and `field` includes the field data. Some tables are joined, since they share the `Soil_ID` column as key.
    """
    )
    return


@app.cell
def _(config, pl):
    proctor = pl.read_csv(config["DATA_PROCTOR"])
    soils = pl.read_csv(config["DATA_SOILS"])
    field = pl.read_csv(config["DATA_FIELD"], null_values="NA")
    data = proctor.join(soils, on="Soil_ID", how="left", coalesce=True)
    data = data.with_columns((pl.col("VolWC_%") / 100).alias("VolWC"))
    return data, field, proctor, soils


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model Training and Diagnostics

    Advanced modeling framework with step-by-step scikit-learn operations, prediction intervals,
    and comprehensive model diagnostics for robust geotechnical analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _(BaseEstimator, TransformerMixin, np, pl):
    class SoilFeatureEngineer(BaseEstimator, TransformerMixin):
        """
        Custom transformer for soil-specific feature engineering.

        Creates interaction terms and domain-specific transformations
        relevant to geotechnical modeling.
        """

        def __init__(self, include_interactions=True):
            self.include_interactions = include_interactions
            self.feature_names_ = None

        def fit(self, X, y=None):
            if hasattr(X, 'columns'):
                self.original_features_ = list(X.columns)
            else:
                self.original_features_ = [f"feature_{i}" for i in range(X.shape[1])]

            self._create_feature_names()
            return self

        def transform(self, X):
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X

            if self.include_interactions and X_array.shape[1] >= 3:
                interactions = []
                if X_array.shape[1] >= 4:  # d85, cu, Gs, Probe
                    interactions.append((X_array[:, 0] * X_array[:, 1]).reshape(-1, 1))  # d85 * cu
                    interactions.append((X_array[:, 2] * X_array[:, 3]).reshape(-1, 1))  # Gs * Probe
                elif X_array.shape[1] == 3:  # d85, cu, Gs
                    interactions.append((X_array[:, 0] * X_array[:, 1]).reshape(-1, 1))  # d85 * cu

                if interactions:
                    X_array = np.hstack([X_array] + interactions)

            return X_array

        def _create_feature_names(self):
            names = self.original_features_.copy()

            if self.include_interactions:
                if len(self.original_features_) >= 4:
                    names.extend(["d85_cu_interaction", "Gs_probe_interaction"])
                elif len(self.original_features_) == 3:
                    names.extend(["d85_cu_interaction"])

            self.feature_names_ = names

        def get_feature_names_out(self, input_features=None):
            return np.array(self.feature_names_)

    def compute_prediction_intervals(model, X, confidence_level=0.95):
        """
        Compute prediction intervals for Gaussian Process models.

        Args:
            model: Fitted Gaussian Process model
            X: Input features
            confidence_level: Confidence level for intervals

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if hasattr(model, 'predict') and hasattr(model, 'predict_y'):
            try:
                predictions, std = model.predict(X, return_std=True)
                alpha = 1 - confidence_level
                z_score = 1.96  # 95% confidence interval

                lower_bounds = predictions - z_score * std
                upper_bounds = predictions + z_score * std

                return predictions, lower_bounds, upper_bounds
            except:
                predictions = model.predict(X)
                return predictions, None, None
        else:
            predictions = model.predict(X)
            return predictions, None, None

    def evaluate_model_diagnostics(y_true, y_pred, model_name="Model"):
        """
        Comprehensive model evaluation with diagnostic metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for reporting

        Returns:
            Dictionary of diagnostic metrics
        """
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        residuals = y_true - y_pred

        diagnostics = {
            'model_name': model_name,
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'residual_skewness': float(pl.Series(residuals).skew()),
            'residual_kurtosis': float(pl.Series(residuals).kurtosis()),
            'n_samples': len(y_true)
        }

        return diagnostics

    def plot_residual_diagnostics(y_true, y_pred, title="Model Diagnostics"):
        """
        Create comprehensive residual diagnostic plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        from scipy import stats

        residuals = y_true - y_pred
        standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs Fitted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)

        # Q-Q Plot
        stats.probplot(standardized_residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True, alpha=0.3)

        # Histogram of Residuals
        ax3.hist(residuals, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)

        # Observed vs Predicted
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax4.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax4.set_xlabel('Observed')
        ax4.set_ylabel('Predicted')
        ax4.set_title('Observed vs Predicted')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        return fig

    def plot_prediction_intervals(X, y_true, y_pred, lower_bounds, upper_bounds,
                                 x_label="Feature", y_label="Target", title="Prediction Intervals"):
        """
        Plot predictions with confidence intervals.

        Args:
            X: Feature values (1D array for plotting)
            y_true: True values
            y_pred: Predicted values
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        if len(X.shape) > 1:
            x_plot = X[:, -1]  # Use last feature (typically Probe)
        else:
            x_plot = X

        sorted_indices = np.argsort(x_plot)
        x_sorted = x_plot[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        if lower_bounds is not None and upper_bounds is not None:
            lower_sorted = lower_bounds[sorted_indices]
            upper_sorted = upper_bounds[sorted_indices]
            ax.fill_between(x_sorted, lower_sorted, upper_sorted, alpha=0.3, color='lightblue', label='95% CI')

        ax.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Prediction')
        ax.scatter(x_plot, y_true, alpha=0.6, color='red', s=20, label='Observations')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def analyze_feature_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
        """
        Analyze feature importance using permutation importance.

        Args:
            model: Fitted model
            X: Feature matrix
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with importance metrics
        """
        from sklearn.inspection import permutation_importance

        if hasattr(model, 'predict'):
            try:
                perm_importance = permutation_importance(
                    model, X, y,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    scoring='r2'
                )

                importance_data = {
                    'feature_names': feature_names,
                    'importance_mean': perm_importance.importances_mean,
                    'importance_std': perm_importance.importances_std,
                }

                # Sort by importance
                sorted_idx = np.argsort(importance_data['importance_mean'])[::-1]
                for key in ['feature_names', 'importance_mean', 'importance_std']:
                    if key == 'feature_names':
                        importance_data[key] = [importance_data[key][i] for i in sorted_idx]
                    else:
                        importance_data[key] = importance_data[key][sorted_idx]

                return importance_data
            except Exception as e:
                print(f"Feature importance analysis failed: {e}")
                return None
        else:
            return None

    def plot_feature_importance(importance_data, title="Feature Importance Analysis"):
        """
        Plot feature importance with error bars.

        Args:
            importance_data: Dictionary from analyze_feature_importance
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        if importance_data is None:
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(importance_data['feature_names']))

        ax.barh(y_pos, importance_data['importance_mean'],
                xerr=importance_data['importance_std'],
                alpha=0.7, color='steelblue', capsize=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_data['feature_names'])
        ax.set_xlabel('Permutation Importance (R² decrease)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def statistical_model_comparison(y_true, y_pred1, y_pred2, model_names=None):
        """
        Perform statistical comparison between two models using paired tests.

        Args:
            y_true: True values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model_names: Names of models for reporting

        Returns:
            Dictionary with test results
        """
        from scipy import stats

        if model_names is None:
            model_names = ["Model 1", "Model 2"]

        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)

        # Paired t-test for mean absolute errors
        t_stat, t_pvalue = stats.ttest_rel(errors1, errors2)

        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_pvalue = stats.wilcoxon(errors1, errors2, alternative='two-sided')

        # Compute effect size (Cohen's d)
        diff = errors1 - errors2
        pooled_std = np.sqrt((np.var(errors1, ddof=1) + np.var(errors2, ddof=1)) / 2)
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0

        results = {
            'model_names': model_names,
            'mean_error_1': np.mean(errors1),
            'mean_error_2': np.mean(errors2),
            'error_difference': np.mean(errors1) - np.mean(errors2),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'cohens_d': cohens_d,
            'sample_size': len(y_true)
        }

        return results

    def sensitivity_analysis(model, X_base, feature_names, perturbation_range=0.1, n_samples=100):
        """
        Perform sensitivity analysis by perturbing input features.

        Args:
            model: Trained model
            X_base: Base feature values for analysis
            feature_names: Names of features
            perturbation_range: Range of perturbation as fraction of feature range
            n_samples: Number of perturbation samples per feature

        Returns:
            Dictionary with sensitivity metrics
        """
        if not hasattr(model, 'predict'):
            return None

        base_prediction = model.predict(X_base.reshape(1, -1))[0]
        sensitivities = {}

        for i, feature_name in enumerate(feature_names):
            feature_range = np.max(X_base) - np.min(X_base)
            perturbations = np.linspace(
                -perturbation_range * feature_range,
                perturbation_range * feature_range,
                n_samples
            )

            predictions = []
            for delta in perturbations:
                X_perturbed = X_base.copy()
                X_perturbed[i] += delta
                pred = model.predict(X_perturbed.reshape(1, -1))[0]
                predictions.append(pred)

            predictions = np.array(predictions)
            sensitivity = np.std(predictions) / np.std(perturbations) if np.std(perturbations) > 0 else 0

            sensitivities[feature_name] = {
                'sensitivity': sensitivity,
                'prediction_range': np.max(predictions) - np.min(predictions),
                'base_prediction': base_prediction,
                'perturbations': perturbations,
                'predictions': predictions
            }

        return sensitivities

    def uncertainty_propagation_analysis(model, X, n_bootstrap=100, confidence_level=0.95):
        """
        Analyze uncertainty propagation through the model using bootstrap resampling.

        Args:
            model: Trained model
            X: Input features
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with uncertainty metrics
        """
        if not hasattr(model, 'predict'):
            return None

        np.random.seed(42)  # For reproducibility
        n_samples = len(X)
        bootstrap_predictions = []

        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_indices]

            try:
                pred_boot = model.predict(X_boot)
                bootstrap_predictions.append(pred_boot)
            except:
                continue

        if not bootstrap_predictions:
            return None

        bootstrap_predictions = np.array(bootstrap_predictions)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        prediction_ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        prediction_ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)

        uncertainty_metrics = {
            'mean_prediction': np.mean(bootstrap_predictions, axis=0),
            'std_prediction': np.std(bootstrap_predictions, axis=0),
            'ci_lower': prediction_ci_lower,
            'ci_upper': prediction_ci_upper,
            'prediction_range': prediction_ci_upper - prediction_ci_lower,
            'confidence_level': confidence_level,
            'n_bootstrap': len(bootstrap_predictions)
        }

        return uncertainty_metrics

    def model_robustness_assessment(model, X, y, noise_levels=[0.01, 0.05, 0.1]):
        """
        Assess model robustness to input noise.

        Args:
            model: Trained model
            X: Input features
            y: True targets
            noise_levels: List of noise standard deviations as fraction of feature std

        Returns:
            Dictionary with robustness metrics
        """
        from sklearn.metrics import r2_score, mean_squared_error

        if not hasattr(model, 'predict'):
            return None

        baseline_pred = model.predict(X)
        baseline_r2 = r2_score(y, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))

        robustness_results = {
            'baseline_r2': baseline_r2,
            'baseline_rmse': baseline_rmse,
            'noise_levels': noise_levels,
            'r2_degradation': [],
            'rmse_increase': []
        }

        for noise_level in noise_levels:
            feature_std = np.std(X, axis=0)
            noise = np.random.normal(0, noise_level * feature_std, X.shape)
            X_noisy = X + noise

            try:
                pred_noisy = model.predict(X_noisy)
                r2_noisy = r2_score(y, pred_noisy)
                rmse_noisy = np.sqrt(mean_squared_error(y, pred_noisy))

                r2_degradation = baseline_r2 - r2_noisy
                rmse_increase = rmse_noisy - baseline_rmse

                robustness_results['r2_degradation'].append(r2_degradation)
                robustness_results['rmse_increase'].append(rmse_increase)
            except:
                robustness_results['r2_degradation'].append(np.nan)
                robustness_results['rmse_increase'].append(np.nan)

        return robustness_results
    return (
        analyze_feature_importance,
        evaluate_model_diagnostics,
        model_robustness_assessment,
        plot_feature_importance,
        plot_residual_diagnostics,
        sensitivity_analysis,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""Overview of data quantity.""")
    return


@app.cell(hide_code=True)
def _(field, proctor, soils):
    print(
        "Number of proctor samples: " + str(proctor["Proctor_ID"].unique().count())
    )
    print("Number of soil samples: " + str(soils["Soil_ID"].unique().count()))
    print("Number of field samples: " + str(field["Field_ID"].unique().count()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model 1: Probe model

    The first model, named *probe model* in the introduction figure, aims at predicting water content using the measure given by the probe and soil three grain parameters $d_{85}$, $Cu$ and specific density $Gs$.

    In 01_multilevel_rosin.ipynb, we discovered that the exponential association wasn't statistically relevant. In this section, we try an exponential association model where each parameter is determined not linearly, as in the bayesian model in 01_multilevel_rosin.ipynb, but with a Gaussian process. We also try to model water content directly as a Gaussian process with features d85, Cu, Gs and the probe value.

    Water content is a variable closed between 0 and 1. To avoid the prediction of negative water contents, or unlikely water contents superior to 100%, the dependant variable can be transformed to a log ratio, such as

    $$
    wlr = log(\frac{θ}{1-θ})
    $$

    These functions transform VWC to WLR, and back.
    """
    )
    return


@app.cell
def _(data, pl, volumetric_water_content_to_log_ratio):
    data_02 = data.with_columns(
        volumetric_water_content_to_log_ratio(pl.col("VolWC")).alias("WLR")
    )
    return (data_02,)


@app.cell
def _(mo):
    mo.md(rf"""Let's see how the probe reacts to water content log ratios.""")
    return


@app.cell
def _(data_02, sns):
    sns.scatterplot(data=data_02, x="Probe", y="WLR", hue="Name")
    return


@app.cell
def _(mo):
    mo.md(
        rf"""
    Points show a growth curve with diminishing return. It could be modelled using an exponential association, typically using three parameters. In this perspective, each test would be fitted to obtain parameters. The parameters could then be subjected to statistical testing and predictive modelling as a function of $g_1$, $g_2$ and $Gs$. Another option is to fit each individual water content (or $wlr$) to $D_{85}$, $Cu$, $Gs$ and the probe value. The result would be less statistically relevant, but very flexible. The last option would also better identify undocumented regions in the data space, and generate more uncertainty in these regions.

    ### Model probe to WLR directly with a Gaussian process

    We use soil grain parameters **and** the probe measurement to predict WLR.
    """
    )
    return


@app.cell
def _(
    RobustScaler,
    config,
    create_stratified_soil_splits,
    data_02,
    split_data_by_soil_ids,
):
    probegp_featureslist = ["d85", "cu", "Gs", "Probe"]
    probegp_targetslist = ["WLR"]
    probegp_df = data_02[["Soil_ID"] + probegp_featureslist + probegp_targetslist + ["Metatype"]].drop_nulls()

    print(f"Probe Model Dataset Analysis:")
    print(f"  Total samples: {probegp_df.shape[0]}")
    print(f"  Unique soils: {probegp_df['Soil_ID'].n_unique()}")

    train_soil_ids, test_soil_ids = create_stratified_soil_splits(
        probegp_df, test_size=0.2, random_state=config["MASTER_SEED"]
    )

    probegp_train_df, probegp_test_df = split_data_by_soil_ids(
        probegp_df, train_soil_ids, test_soil_ids
    )

    print(f"  Training soils: {len(train_soil_ids)}, samples: {probegp_train_df.shape[0]}")
    print(f"  Test soils: {len(test_soil_ids)}, samples: {probegp_test_df.shape[0]}")

    probegp_features = probegp_train_df.select(probegp_featureslist).to_numpy()
    probegp_targets = probegp_train_df.select(probegp_targetslist).to_numpy()

    probegp_featuresScaler = RobustScaler()
    probegp_featuressc = probegp_featuresScaler.fit_transform(probegp_features)
    return (
        probegp_featuresScaler,
        probegp_featureslist,
        probegp_featuressc,
        probegp_targets,
        probegp_test_df,
        probegp_train_df,
    )


@app.cell
def _(mo):
    mo.md(rf"""Optuna is optimizing Gaussian Process hyperparameters (alpha, nu, length_scale) by running multiple trials, evaluating each combination through cross-validation, and progressively converging toward the configuration that minimizes prediction error while controlling for overfitting.""")
    return


@app.cell
def _(
    config,
    mo,
    optimize_gaussian_process_hyperparameters,
    probegp_featuressc,
    probegp_targets,
    probegp_train_df,
):
    # Extract soil IDs for group-based cross-validation
    with mo.persistent_cache(name="probe_best_params_cache"): # caching long computations
        probegp_soil_ids = probegp_train_df["Soil_ID"].to_numpy()

        probe_best_params = optimize_gaussian_process_hyperparameters(
            features=probegp_featuressc,
            targets=probegp_targets,
            config=config,
            study_name="Probe GP Hyperparameter Optimization",
            random_seed=config["PROBE_MODEL_SEED"],
            soil_ids=probegp_soil_ids
        )
        probe_best_params
    return (probe_best_params,)


@app.cell
def _(
    GaussianProcessRegressor,
    GroupKFold,
    Matern,
    WhiteKernel,
    config,
    cross_val_score,
    evaluate_model_diagnostics,
    log_ratio_to_volumetric_water_content,
    mo,
    np,
    probe_best_params,
    probegp_featuresScaler,
    probegp_featureslist,
    probegp_test_df,
    probegp_train_df,
    rmse_vwc_scorer,
):
    np.random.seed(config["PROBE_MODEL_SEED"] + 1000)

    with mo.persistent_cache(name="probe_model_cache"): 
        probegp_model = GaussianProcessRegressor(
            kernel=Matern(
                length_scale=probe_best_params["length_scale"],
                nu=probe_best_params["nu"],
                length_scale_bounds=config["GP_LENGTH_SCALE_BOUNDS"]
            ) + WhiteKernel(noise_level=0.1),
            n_restarts_optimizer=config["N_RESTARTS_OPTIMIZER"],
            alpha=probe_best_params["alpha"],
            optimizer="fmin_l_bfgs_b",
            normalize_y=True,
            copy_X_train=False,
            random_state=config["PROBE_MODEL_SEED"] + 2000,
        )

        soil_groups = probegp_train_df["Soil_ID"].to_numpy()
        probegp_cv = GroupKFold(n_splits=len(np.unique(soil_groups)))

        X_train = probegp_train_df.select(probegp_featureslist).to_numpy()
        y_train = probegp_train_df.select(["WLR"]).to_numpy().ravel()
        X_train_scaled = probegp_featuresScaler.transform(X_train)

        probegp_rmse_scores = -cross_val_score(
            probegp_model,
            X_train_scaled,
            y_train,
            cv=probegp_cv,
            groups=soil_groups,
            scoring=rmse_vwc_scorer
        )

        probegp_r2_scores = cross_val_score(
            probegp_model,
            X_train_scaled,
            y_train,
            cv=probegp_cv,
            groups=soil_groups,
            scoring="r2",
        )

        print("X_train shape:", X_train.shape)
        print("First row of X_train:", X_train[0])

        probegp_model.fit(X_train_scaled, y_train)

        X_test = probegp_test_df.select(probegp_featureslist).to_numpy()
        y_test = probegp_test_df.select(["WLR"]).to_numpy().ravel()
        X_test_scaled = probegp_featuresScaler.transform(X_test)

        y_pred_test = probegp_model.predict(X_test_scaled)

        probe_test_diagnostics = evaluate_model_diagnostics(
            y_test, y_pred_test, "Probe Model (Test Set)"
        )

        y_pred_test_vwc = log_ratio_to_volumetric_water_content(y_pred_test)
        y_test_vwc = log_ratio_to_volumetric_water_content(y_test)

        probe_test_diagnostics_vwc = evaluate_model_diagnostics(
            y_test_vwc, y_pred_test_vwc, "Probe Model VWC (Test Set)"
        )

    print(f"Probe Model Test Set Performance:")
    print(f"  WLR Domain - R²: {probe_test_diagnostics['r2']:.4f}, "
          f"RMSE: {probe_test_diagnostics['rmse']:.4f}")
    print(f"  VWC Domain - R²: {probe_test_diagnostics_vwc['r2']:.4f}, "
          f"RMSE: {probe_test_diagnostics_vwc['rmse']:.4f}")
    print(f"Probe Model Cross-Validation Results:")
    print(f"  RMSE (VWC) - Min: {np.min(probegp_rmse_scores):.4f}, "
          f"Median: {np.median(probegp_rmse_scores):.4f}, "
          f"Max: {np.max(probegp_rmse_scores):.4f}")
    print(f"  R² - Min: {np.min(probegp_r2_scores):.4f}, "
          f"Median: {np.median(probegp_r2_scores):.4f}, "
          f"Max: {np.max(probegp_r2_scores):.4f}")
    return (probegp_model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model Diagnostics and Residual Analysis

    Comprehensive evaluation of model performance including residual analysis,
    prediction intervals, and statistical diagnostics to assess model reliability.
    """
    )
    return


@app.cell
def _(
    log_ratio_to_volumetric_water_content,
    plot_residual_diagnostics,
    probegp_featuresScaler,
    probegp_featureslist,
    probegp_model,
    probegp_train_df,
):
    X_train_diag = probegp_train_df.select(probegp_featureslist).to_numpy()
    y_train_diag = probegp_train_df.select(["WLR"]).to_numpy().ravel()
    X_train_diag_scaled = probegp_featuresScaler.transform(X_train_diag)

    y_pred_train_diag = probegp_model.predict(X_train_diag_scaled)

    y_train_vwc_diag = log_ratio_to_volumetric_water_content(y_train_diag)
    y_pred_vwc_diag = log_ratio_to_volumetric_water_content(y_pred_train_diag)

    probe_residual_fig = plot_residual_diagnostics(
        y_train_vwc_diag, y_pred_vwc_diag,
        title="Probe Model Residual Diagnostics (VWC Domain)"
    )
    probe_residual_fig.savefig("images/probe_model_diagnostics.png", dpi=300, bbox_inches='tight')
    probe_residual_fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Advanced Model Analysis and Statistical Testing

    Comprehensive analysis including feature importance, sensitivity analysis,
    model robustness assessment, and statistical significance testing for
    rigorous evaluation of geotechnical modeling performance.
    """
    )
    return


@app.cell
def _(probegp_train_df):
    probegp_train_df
    return


@app.cell
def _(
    analyze_feature_importance,
    config,
    plot_feature_importance,
    probegp_featureslist,
    probegp_model,
    probegp_train_df,
):
    X_probe_importance = probegp_train_df.select(probegp_featureslist).to_numpy()
    y_probe_importance = probegp_train_df.select(["WLR"]).to_numpy().ravel()

    probe_importance_data = analyze_feature_importance(
        probegp_model,
        X_probe_importance,
        y_probe_importance,
        probegp_featureslist,
        n_repeats=10,
        random_state=config["MASTER_SEED"]
    )

    print("probe_importance_data: ", probe_importance_data)

    probe_importance_fig = plot_feature_importance(
        probe_importance_data,
        title="Probe Model Feature Importance Analysis"
    )
    probe_importance_fig.savefig("images/probe_feature_importance.png", dpi=300, bbox_inches='tight')
    probe_importance_fig.show()

    print("Probe Model Feature Importance Rankings:")
    for _i, (_name, _importance, _std) in enumerate(zip(
        probe_importance_data['feature_names'],
        probe_importance_data['importance_mean'],
        probe_importance_data['importance_std']
    )):
        print(f"  {_i+1}. {_name}: {_importance:.4f} ± {_std:.4f}")
    return


@app.cell
def _(
    model_robustness_assessment,
    probegp_featureslist,
    probegp_model,
    probegp_train_df,
):
    X_probe_robust = probegp_train_df.select(probegp_featureslist).to_numpy()
    y_probe_robust = probegp_train_df.select(["WLR"]).to_numpy().ravel()

    probe_robustness = model_robustness_assessment(
        probegp_model,
        X_probe_robust,
        y_probe_robust,
        noise_levels=[0.01, 0.05, 0.1, 0.2]
    )

    print("Probe Model Robustness Assessment:")
    print(f"  Baseline R²: {probe_robustness['baseline_r2']:.4f}")
    print(f"  Baseline RMSE: {probe_robustness['baseline_rmse']:.4f}")
    print("  Noise Level → R² Degradation | RMSE Increase")
    for _noise, _r2_deg, _rmse_inc in zip(
        probe_robustness['noise_levels'],
        probe_robustness['r2_degradation'],
        probe_robustness['rmse_increase']
    ):
        print(f"    {_noise:5.1%} → {_r2_deg:13.4f} | {_rmse_inc:12.4f}")
    return


@app.cell
def _(
    np,
    probegp_featureslist,
    probegp_model,
    probegp_train_df,
    sensitivity_analysis,
):
    X_probe_sens = probegp_train_df.select(probegp_featureslist).to_numpy()

    representative_sample = np.median(X_probe_sens, axis=0)

    probe_sensitivity = sensitivity_analysis(
        probegp_model,
        representative_sample,
        probegp_featureslist,
        perturbation_range=0.1,
        n_samples=50
    )

    print("Probe Model Sensitivity Analysis:")
    print("  Feature → Sensitivity | Prediction Range")
    for feature_name in probegp_featureslist:
        sens_data = probe_sensitivity[feature_name]
        print(f"  {feature_name:8} → {sens_data['sensitivity']:10.4f} | {sens_data['prediction_range']:15.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""Fit the model and generate predictions.""")
    return


@app.cell
def _(probegp_featuressc, probegp_model, probegp_targets):
    probegp_model.fit(probegp_featuressc, probegp_targets)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```
     	kernel  	Matern(length...ise_level=0.1)
    	alpha  	0.011354706203833605
    	optimizer  	'fmin_l_bfgs_b'
    	n_restarts_optimizer  	5
    	normalize_y  	True
    	copy_X_train  	False
    	n_targets  	None
    	random_state  	910493
    	kernel__k1  	Matern(length....326, nu=2.49)
    	kernel__k2  	WhiteKernel(noise_level=0.1)
    	kernel__k1__length_scale  	0.3255451000840303
    	kernel__k1__length_scale_bounds  	(0.01, ...)
    	kernel__k1__nu  	2.493908922993146
    	kernel__k2__noise_level  	0.1
    	kernel__k2__noise_level_bounds  	(1e-05, ...)
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(rf"""Plot model 1.""")
    return


@app.cell
def _(probegp_train_df):
    probegp_train_df
    return


@app.cell
def _(
    log_ratio_to_volumetric_water_content,
    pl,
    probegp_featuressc,
    probegp_model,
    probegp_targets,
    probegp_train_df,
    sns,
):
    probegp_model_targets_pred = log_ratio_to_volumetric_water_content(
        probegp_model.predict(probegp_featuressc)
    )
    probegp_model_targets_obs = log_ratio_to_volumetric_water_content(probegp_targets.flatten())
    plot_data = pl.DataFrame(
        {
            "Observed θ in Proctor testing": probegp_model_targets_obs,
            "Predicted θ with GP modelling": probegp_model_targets_pred,
            "Soil type": probegp_train_df["Metatype"],
        }
    )
    abline = [0, 0.35]
    g = sns.jointplot(
        data=plot_data,
        x="Observed θ in Proctor testing",
        y="Predicted θ with GP modelling",
        hue="Soil type",
        palette="gray",
    )
    g.ax_joint.plot(abline, abline, "black")
    g.savefig("images/probemodel-obs-pred.png")
    g
    return


@app.cell
def _(mo):
    mo.md(rf"""Plot all calibration curves.""")
    return


@app.cell
def _(
    data_02,
    log_ratio_to_volumetric_water_content,
    np,
    pl,
    probegp_featuresScaler,
    probegp_featureslist,
    probegp_model,
    soils,
):
    soil_ids = soils["Soil_ID"].unique().to_list()
    soil_ids_list = []
    gs_list = []
    d85_list = []
    cu_list = []
    probe_list = []
    vwc_list = []
    n_probe = 100
    probe = np.linspace(1800, 2800, n_probe)

    for k, soil_id_k in enumerate(soil_ids):
        data_02_idfeatures_i = (
            data_02.filter(pl.col("Soil_ID") == soil_id_k)
            .select(probegp_featureslist)
            .to_numpy()
        )
        data_02_idtargets_i = (
            data_02.filter(pl.col("Soil_ID") == soil_id_k)
            .select("VolWC")
            .to_numpy()
        )

        probegp_model_predfeatures = np.vstack(
            [data_02_idfeatures_i[0, :]] * n_probe
        )
        probegp_model_predfeatures[:, -1] = probe
        probegp_model_predfeatures_sc = probegp_featuresScaler.transform(
            probegp_model_predfeatures
        )
        probegp_model_predtargets = log_ratio_to_volumetric_water_content(
            probegp_model.predict(probegp_model_predfeatures_sc)
        )

        soil_ids_list.extend([soil_id_k] * n_probe)
        gs_list.extend([data_02_idfeatures_i[0, 2]] * n_probe)
        d85_list.extend([data_02_idfeatures_i[0, 1]] * n_probe)
        cu_list.extend([data_02_idfeatures_i[0, 0]] * n_probe)
        probe_list.extend(probe)
        vwc_list.extend(probegp_model_predtargets)

    # Créez le DataFrame Polars à partir des listes.
    model1_preds = pl.DataFrame(
        {
            "Soil_ID": soil_ids_list,
            "Gs": gs_list,
            "d85": d85_list,
            "Cu": cu_list,
            "Probe": probe_list,
            "VWC": vwc_list,
        }
    )
    model1_preds
    return model1_preds, soil_ids


@app.cell
def _(data_02, model1_preds, np, pl, plt, probegp_featureslist, soil_ids):
    data_02_features = data_02.select(probegp_featureslist).to_numpy()
    data_02_targets = data_02.select("VolWC").to_numpy()
    plot_nrow = 8
    plot_ncol = 4

    fig, axs = plt.subplots(plot_nrow, plot_ncol, figsize=(16, 24))
    for m, soil_id in enumerate(soil_ids):
        data_02_idfeatures = (
            data_02.filter(pl.col("Soil_ID") == soil_id)
            .select(probegp_featureslist)
            .to_numpy()
        )
        data_02_idtargets = (
            data_02.filter(pl.col("Soil_ID") == soil_id).select("VolWC").to_numpy()
        )
        probe_m = (
            model1_preds.filter(pl.col("Soil_ID") == soil_id)
            .select("Probe")
            .to_numpy()
        )
        vwc = (
            model1_preds.filter(pl.col("Soil_ID") == soil_id)
            .select("VWC")
            .to_numpy()
        )
        row = m // plot_ncol
        col = m % plot_ncol
        axs[row, col].plot(
            data_02_features[:, 3], data_02_targets, ".", color="black", alpha=0.1
        )  # plot all data_02
        axs[row, col].plot(probe_m, vwc, color="k")  # plot model predictions
        axs[row, col].plot(
            data_02_idfeatures[:, 3], data_02_idtargets, "o", color="black"
        )  # plot data_02 for soil_id
        axs[row, col].text(
            1800,
            0.32,
            "Gs=" + str(np.round(data_02_idfeatures[0, 2], 2)) + "g/cm³",
        )
        axs[row, col].text(
            1800,
            0.29,
            "d$_{85}$=" + str(np.round(data_02_idfeatures[0, 1], 2)) + "mm",
        )
        axs[row, col].text(
            1800, 0.26, "Cu=" + str(np.round(data_02_idfeatures[0, 0], 2))
        )
        axs[row, col].set_title("Soil " + str(soil_id))
        if col == 0:  # Only add y label for the first column
            axs[row, col].set_ylabel("Volumetric Water Content")
        else:  # Hide y ticks for other columns
            axs[row, col].tick_params(left=False, labelleft=False)
        if row == plot_nrow - 1 or (
            row == plot_nrow - 2 and col == plot_ncol - 1
        ):  # Only add x label for the last row or the last plot
            axs[row, col].set_xlabel("Probe")
            axs[row, col].tick_params(bottom=True, labelbottom=True)
        else:  # Hide x ticks for other rows
            axs[row, col].tick_params(bottom=False, labelbottom=False)
        axs[row, col].set_ylim(0, 0.35)


    # Hide remaining axes
    for n in range(m + 1, plot_nrow * plot_ncol):
        row = n // plot_ncol
        col = n % plot_ncol
        axs[row, col].axis("off")

    plt.tight_layout()
    fig.savefig("images/probemodel-fitted.png")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model 2: $ρ_d$ model

    The final objective is to obtain $ρ_d$. To obtain it, we need to know the degree of saturation... which we do not have since, to compute it, we need $ρ_d$. To extricate ourselves to this tautological loop, one strategy is to take two water contents: one measurement in the soil in situ conditions, and a second measurement after a water column has been applied to the soil over 1 minute. The soil wouldn't be saturated then. And its degree of saturation will depend on many factors: initial water content, grain-size, Gs and $ρ_d$, which... we don't have. Fortunately, a foresighted degree of saturation can be approximated from the initial water content, $g_1$, $g_2$ and $Gs$, with the use of machine learning. Having a measured volumetric water content, a foresighted degree of saturation, $Gs$ and $ρ_w$, computing $ρ_d$ is a simple phase transformation.

    $$ρ_d = (1 - \frac{θ}{Sr}) \times Gs \times ρ_w$$

    And porosity can be found with

    $$ϕ = 1 - \frac{ρ_d}{Gs ρ_w}$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(rf"""Data from table `field`, containing results from field experiments, will be used. We'll need $d_{85}$ and $Cu$, which are joined to the `field` table.""")
    return


@app.cell
def _(field, soils):
    field_psd = field.join(
        soils.select(["Soil_ID", "cu", "d85"]),
        on="Soil_ID",
        how="left",
        coalesce=False,
    )
    return (field_psd,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Just like volumetric water content, degree of saturation is limited to span between 0 and 1. To preprocess it, we transformed it to a degree of saturation log-ratio equivalent to the log volume of water divided by its complementary, volume of air.

    $$
    Srlr = ln(\frac{Sr}{1-Sr})
    $$

    $$
    Sr = \frac{e^{Srlr}}{1+e^{Srlr}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""We create a new data frame, containing the columns we need, then removing all rows containing at least one null (unmeasured) value.""")
    return


@app.cell
def _(field_psd):
    field_full = field_psd.select(
        [
            "Soil_ID",
            "d85",
            "cu",
            "Gs",
            "Probe_before_flood",
            "Probe_after_1minflood",
            "Density_kg/m3_PT",
            "Metatype",
        ]
    ).drop_nulls()
    return (field_full,)


@app.cell
def _(mo):
    mo.md(r"""Predict θ_R1 and θ_R2 with one of the models created before.""")
    return


@app.cell
def _(field_full):
    probefield_R1 = field_full.select("Probe_before_flood").to_numpy()
    probefield_R2 = field_full.select("Probe_after_1minflood").to_numpy()
    return


@app.cell
def _(field_full, probegp_featuresScaler, probegp_model):
    probegp_model_fieldfeatureslist = ["d85", "cu", "Gs"]
    probegp_model_fieldfeatures_before = field_full.select(
        probegp_model_fieldfeatureslist + ["Probe_before_flood"]
    ).to_numpy()
    probegp_model_fieldfeatures_after = field_full.select(
        probegp_model_fieldfeatureslist + ["Probe_after_1minflood"]
    ).to_numpy()
    probegp_model_fieldfeatures_beforesc = probegp_featuresScaler.transform(
        probegp_model_fieldfeatures_before
    )
    probegp_model_fieldfeatures_aftersc = probegp_featuresScaler.transform(
        probegp_model_fieldfeatures_after
    )
    WLR_R1 = probegp_model.predict(probegp_model_fieldfeatures_beforesc)
    WLR_R2 = probegp_model.predict(probegp_model_fieldfeatures_aftersc)
    return WLR_R1, WLR_R2


@app.cell
def _(mo):
    mo.md(r"""Gather the info for modelling.""")
    return


@app.cell
def _(
    WLR_R1,
    WLR_R2,
    config,
    field_full,
    log_ratio_to_volumetric_water_content,
    phase_Sr,
    pl,
    porosity,
    saturation_ratio_to_log_ratio,
):
    θ_R1 = log_ratio_to_volumetric_water_content(WLR_R1)
    θ_R2 = log_ratio_to_volumetric_water_content(WLR_R2)

    field_results = (
        field_full
        .with_columns(
            porosity(
                rho_d=pl.col("Density_kg/m3_PT"),
                rho_w=config["WATER_DENSITY"],
                gs=pl.col("Gs")
            ).alias("porosity")
        )
        .with_columns(
            [
                pl.Series(WLR_R1).alias("WLR_R1"),
                pl.Series(WLR_R2).alias("WLR_R2"),
                pl.Series(θ_R1).alias("θ_R1"),
                pl.Series(θ_R2).alias("θ_R2"),
                pl.lit(config["WATER_DENSITY"]).alias("ρw"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("θ_R1") < pl.col("porosity"))
                .then(pl.col("θ_R1"))
                .otherwise(pl.col("porosity") * config["MAX_SATURATION"]),
                pl.when(pl.col("θ_R2") < pl.col("porosity"))
                .then(pl.col("θ_R2"))
                .otherwise(pl.col("porosity") * config["MAX_SATURATION"]),
            ]
        )
        .with_columns(
            phase_Sr(
                theta=pl.col("θ_R2"),
                rho_d=pl.col("Density_kg/m3_PT"),
                rho_w=pl.col("ρw"),
                gs=pl.col("Gs"),
            ).alias("Sr_R2")
        )
        .with_columns(saturation_ratio_to_log_ratio(pl.col("Sr_R2")).alias("SrLR_R2"))
    )
    return (field_results,)


@app.cell
def _(mo):
    mo.md(r"""Transforming data.""")
    return


@app.cell
def _(
    RobustScaler,
    config,
    create_stratified_soil_splits,
    field_results,
    pl,
    split_data_by_soil_ids,
):
    srmod_features = ["WLR_R1", "d85", "cu", "Gs"]
    srmod_target = ["SrLR_R2"]
    srmod_df = (
        field_results.select(
            srmod_features
            + srmod_target
            + ["Density_kg/m3_PT", "θ_R2", "Metatype", "Soil_ID"]
        )
        .drop_nulls()
        .filter(~pl.col("SrLR_R2").is_nan())
    )

    print(f"Saturation Ratio Model Dataset Analysis:")
    print(f"  Total samples: {srmod_df.shape[0]}")
    print(f"  Unique soils: {srmod_df['Soil_ID'].n_unique()}")

    sr_train_soil_ids, sr_test_soil_ids = create_stratified_soil_splits(
        srmod_df, test_size=0.2, random_state=config["MASTER_SEED"] + 100
    )

    srmod_train_df, srmod_test_df = split_data_by_soil_ids(
        srmod_df, sr_train_soil_ids, sr_test_soil_ids
    )

    print(f"  Training soils: {len(sr_train_soil_ids)}, samples: {srmod_train_df.shape[0]}")
    print(f"  Test soils: {len(sr_test_soil_ids)}, samples: {srmod_test_df.shape[0]}")

    srmod_featurestr = srmod_train_df[srmod_features].to_numpy()
    srmod_targettr = srmod_train_df[srmod_target].to_numpy()

    srmod_featuresScaler = RobustScaler()
    srmod_featuressc = srmod_featuresScaler.fit_transform(srmod_featurestr)
    return (
        sr_train_soil_ids,
        srmod_df,
        srmod_features,
        srmod_featuresScaler,
        srmod_featuressc,
        srmod_targettr,
        srmod_test_df,
        srmod_train_df,
    )


@app.cell
def _(mo):
    mo.md(r"""Looking for the most appropriate `alpha` with an Optuna search cross validation.""")
    return


@app.cell
def _(
    config,
    mo,
    optimize_gaussian_process_hyperparameters,
    srmod_featuressc,
    srmod_targettr,
    srmod_train_df,
):
    # Extract soil IDs for group-based cross-validation
    srmod_soil_ids = srmod_train_df["Soil_ID"].to_numpy()

    with mo.persistent_cache(name="sr_best_params_cache"):
        sr_best_params = optimize_gaussian_process_hyperparameters(
            features=srmod_featuressc,
            targets=srmod_targettr,
            config=config,
            study_name="Saturation Ratio GP Hyperparameter Optimization",
            random_seed=config["SR_MODEL_SEED"],
            soil_ids=srmod_soil_ids
        )
        sr_best_params
    return (sr_best_params,)


@app.cell
def _(
    GaussianProcessRegressor,
    GroupKFold,
    Matern,
    WhiteKernel,
    config,
    cross_val_score,
    evaluate_model_diagnostics,
    log_ratio_to_saturation_ratio,
    mo,
    np,
    sr_best_params,
    sr_train_soil_ids,
    srmod_features,
    srmod_featuresScaler,
    srmod_featuressc,
    srmod_targettr,
    srmod_test_df,
    srmod_train_df,
):
    np.random.seed(config["SR_MODEL_SEED"] + 1000)

    with mo.persistent_cache(name="sr_model_cache"):
        srlr_base_model = GaussianProcessRegressor(
            kernel=Matern(
                length_scale=sr_best_params["length_scale"],
                nu=sr_best_params["nu"],
                length_scale_bounds=config["GP_LENGTH_SCALE_BOUNDS"]
            ) + WhiteKernel(noise_level=0.1),
            n_restarts_optimizer=config["N_RESTARTS_OPTIMIZER"],
            alpha=sr_best_params["alpha"],
            normalize_y=True,
            random_state=config["SR_MODEL_SEED"] + 2000,
        )

        srlr_model = srlr_base_model

        srlr_cv = GroupKFold(n_splits=len(sr_train_soil_ids))

        X_train_sr = srmod_featuressc
        y_train_sr = srmod_targettr

        rmse_scores_sr = -cross_val_score(
            srlr_model,
            X_train_sr,
            y_train_sr,
            cv=srlr_cv,
            groups=srmod_train_df["Soil_ID"].to_numpy(),
            scoring="neg_root_mean_squared_error",
        )

        r2_score_srlr = cross_val_score(
            srlr_model,
            X_train_sr,
            y_train_sr,
            cv=srlr_cv,
            groups=srmod_train_df["Soil_ID"].to_numpy(),
            scoring="r2",
        )

        srlr_model.fit(X_train_sr, y_train_sr)


        X_test_sr_raw = srmod_test_df.select(srmod_features).to_numpy()
        X_test_sr = srmod_featuresScaler.transform(X_test_sr_raw)
        y_test_sr = srmod_test_df.select(["SrLR_R2"]).to_numpy().ravel()

        y_pred_test_sr = srlr_model.predict(X_test_sr)

        sr_test_diagnostics = evaluate_model_diagnostics(
            y_test_sr, y_pred_test_sr, "Saturation Ratio Model (Test Set)"
        )

        y_pred_test_sr_ratio = log_ratio_to_saturation_ratio(y_pred_test_sr)
        y_test_sr_ratio = log_ratio_to_saturation_ratio(y_test_sr)

        sr_test_diagnostics_ratio = evaluate_model_diagnostics(
            y_test_sr_ratio, y_pred_test_sr_ratio, "Saturation Ratio Model Sr (Test Set)"
        )

    print(f"Saturation Ratio Model Test Set Performance:")
    print(f"  SrLR Domain - R²: {sr_test_diagnostics['r2']:.4f}, "
          f"RMSE: {sr_test_diagnostics['rmse']:.4f}")
    print(f"  Sr Domain - R²: {sr_test_diagnostics_ratio['r2']:.4f}, "
          f"RMSE: {sr_test_diagnostics_ratio['rmse']:.4f}")
    print(f"Saturation Ratio Model Cross-Validation Results:")
    print(f"  RMSE (Sr) - Min: {np.min(rmse_scores_sr):.4f}, "
          f"Median: {np.median(rmse_scores_sr):.4f}, "
          f"Max: {np.max(rmse_scores_sr):.4f}")
    print(f"  R² - Min: {np.min(r2_score_srlr):.4f}, "
          f"Median: {np.median(r2_score_srlr):.4f}, "
          f"Max: {np.max(r2_score_srlr):.4f}")
    return (srlr_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Saturation Model Advanced Analysis

    Comprehensive evaluation of the saturation ratio model including feature importance,
    sensitivity analysis, and robustness assessment for the geotechnical prediction pipeline.
    """
    )
    return


@app.cell
def _(
    analyze_feature_importance,
    config,
    plot_feature_importance,
    srlr_model,
    srmod_features,
    srmod_featuressc,
    srmod_targettr,
):
    X_sr_importance = srmod_featuressc
    y_sr_importance = srmod_targettr

    sr_importance_data = analyze_feature_importance(
        srlr_model,
        X_sr_importance,
        y_sr_importance,
        srmod_features,
        n_repeats=10,
        random_state=config["MASTER_SEED"] + 200
    )
    sr_importance_fig = plot_feature_importance(
        sr_importance_data,
        title="Saturation Ratio Model Feature Importance Analysis"
    )
    sr_importance_fig.savefig("images/sr_feature_importance.png", dpi=300, bbox_inches='tight')
    sr_importance_fig.show()

    print("Saturation Ratio Model Feature Importance Rankings:")
    for _i, (_name, _importance, _std) in enumerate(zip(
        sr_importance_data['feature_names'],
        sr_importance_data['importance_mean'],
        sr_importance_data['importance_std']
    )):
        print(f"  {_i+1}. {_name}: {_importance:.4f} ± {_std:.4f}")
    return


@app.cell
def _(
    model_robustness_assessment,
    srlr_model,
    srmod_featuressc,
    srmod_targettr,
):
    X_sr_robust = srmod_featuressc
    y_sr_robust = srmod_targettr

    sr_robustness = model_robustness_assessment(
        srlr_model,
        X_sr_robust,
        y_sr_robust,
        noise_levels=[0.01, 0.05, 0.1, 0.2]
    )

    print("Saturation Ratio Model Robustness Assessment:")
    print(f"  Baseline R²: {sr_robustness['baseline_r2']:.4f}")
    print(f"  Baseline RMSE: {sr_robustness['baseline_rmse']:.4f}")
    print("  Noise Level → R² Degradation | RMSE Increase")
    for _noise, _r2_deg, _rmse_inc in zip(
        sr_robustness['noise_levels'],
        sr_robustness['r2_degradation'],
        sr_robustness['rmse_increase']
    ):
        print(f"    {_noise:5.1%} → {_r2_deg:13.4f} | {_rmse_inc:12.4f}")
    return


@app.cell
def _(
    log_ratio_to_saturation_ratio,
    root_mean_squared_error,
    srlr_model,
    srmod_featuressc,
    srmod_targettr,
):
    srlr_model.fit(srmod_featuressc, srmod_targettr)
    srlr_pred = srlr_model.predict(srmod_featuressc)

    print(f"Saturation Ratio Model Training Results:")
    print(f"  RMSE: {root_mean_squared_error(srmod_targettr, srlr_pred):.4f}")

    sr_pred = log_ratio_to_saturation_ratio(srlr_pred)
    return (sr_pred,)


@app.cell
def _(log_ratio_to_saturation_ratio, pl, sns, sr_pred, srmod_train_df):
    plot_data_sr = pl.DataFrame(
        {
            "Observed Sr in field testing": log_ratio_to_saturation_ratio(
                srmod_train_df["SrLR_R2"].to_numpy()
            ),
            "Predicted Sr with GP modelling": sr_pred,
            "Soil type": srmod_train_df["Metatype"],
        }
    )
    abline_sr = [0.4, 1.0]
    gsr = sns.jointplot(
        data=plot_data_sr,
        x="Observed Sr in field testing",
        y="Predicted Sr with GP modelling",
        hue="Soil type",
        palette="gray",
        xlim=[0.3, 1.1],
        ylim=[0.3, 1.1],
    )
    gsr.ax_joint.plot(abline_sr, abline_sr, "black")
    gsr.savefig("images/srmodel-obs-pred.png")
    gsr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""Once Sr is modelled, predictions can be expressed in terms of $ρ_d$.""")
    return


@app.cell
def _(config, phase_ρd, pl, sns, sr_pred, srmod_train_df):
    plot_data_ρd = pl.DataFrame(
        {
            "Observed ρd (kg/m³)": srmod_train_df["Density_kg/m3_PT"],
            "Predicted ρd (kg/m³)": phase_ρd(
                theta=srmod_train_df["θ_R2"],
                sr=sr_pred,
                rho_w=config["WATER_DENSITY"],
                gs=srmod_train_df["Gs"]
            ),
            "Soil type": srmod_train_df["Metatype"],
        }
    )

    abline_ρd = [1600, 2400]
    gρd = sns.jointplot(
        data=plot_data_ρd,
        x="Observed ρd (kg/m³)",
        y="Predicted ρd (kg/m³)",
        hue="Soil type",
        palette="gray",
    )
    gρd.ax_joint.plot(abline_ρd, abline_ρd, "black")
    gρd.fig.suptitle("(a) Sherbrooke Method", x=0.0, y=0.975, ha="left", va="top")
    gρd.savefig("images/rhodmodel-obs-pred.png")
    gρd
    return abline_ρd, plot_data_ρd


@app.cell
def _(pl, plot_data_ρd):
    predρd_stats = plot_data_ρd.with_columns(
        (
            (pl.col("Observed ρd (kg/m³)") - plot_data_ρd["Predicted ρd (kg/m³)"])
            ** 2
        ).alias("squared_errors")
    ).with_columns(pl.Series("Device", ["Sherbrooke Method"] * len(plot_data_ρd)))
    return (predρd_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compared to nucleodensimeter...""")
    return


@app.cell
def _(abline_ρd, field, pd, sns):
    plot_data_ND = pd.DataFrame(
        {
            "Observed ρd (kg/m³)": field["Density_kg/m3_PT"],
            "Predicted ρd (kg/m³)": field["Density_kg/m3_ND"],
            "Soil type": field["Metatype"],
        }
    )

    abline_ND = [1600, 2400]
    gND = sns.jointplot(
        data=plot_data_ND,
        x="Observed ρd (kg/m³)",
        y="Predicted ρd (kg/m³)",
        hue="Soil type",
        palette="gray",
    )
    gND.ax_joint.plot(abline_ND, abline_ρd, "black")
    gND.fig.suptitle("(b) Nucleodensimeter", x=0.0, y=0.975, ha="left", va="top")
    gND.savefig("images/ND-obs-pred.png")
    gND
    return (plot_data_ND,)


@app.cell
def _(pl, plot_data_ND):
    plot_data_ND_pl = pl.DataFrame(plot_data_ND)
    predND_stats = (
        plot_data_ND_pl
        .with_columns(
            ((pl.col("Observed ρd (kg/m³)") - plot_data_ND_pl["Predicted ρd (kg/m³)"]) ** 2).alias("squared_errors")
        )
        .with_columns(pl.Series("Device", ["Nucleodensimeter"] * len(plot_data_ND_pl)))
    )
    return (predND_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Summary statistics for accuracy assessment.""")
    return


@app.cell
def _(config, np, pl, predND_stats, predρd_stats):
    pred_stats = predρd_stats.drop_nulls().vstack(predND_stats.drop_nulls())
    pred_stats_per_soil = (
        pred_stats.group_by(["Device", "Soil type"])
        .agg(pl.mean("squared_errors").alias("mean_squared_error"))
        .with_columns(np.sqrt(pl.col("mean_squared_error")).alias("RMSE"))
        .drop("mean_squared_error")
    )
    pred_stats_all = (
        pred_stats.group_by("Device")
        .agg(pl.mean("squared_errors").alias("mean_squared_error"))
        .with_columns(pl.Series("Soil type", ["MG20, MG112"] * 2))
        .with_columns(np.sqrt(pl.col("mean_squared_error")).alias("RMSE"))
        .drop("mean_squared_error")
    )
    pred_stats_summary = pred_stats_per_soil.vstack(pred_stats_all)
    pred_stats_summary.write_csv(config["OUTPUT_PRED_STATS"])

    print(pred_stats_summary)
    return (pred_stats_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bootstrap Confidence Intervals and Leave-One-Soil-Out Cross-Validation

    This section implements bootstrap analysis with block resampling by `Soil_ID` and leave-one-soil-out cross-validation to assess model uncertainty and generalization performance. We first preparethe functions for bootstrap RMSE and LOOCV RMSE.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    GaussianProcessRegressor,
    Matern,
    RobustScaler,
    WhiteKernel,
    config,
    data_02,
    field_results,
    log_ratio_to_saturation_ratio,
    np,
    phase_ρd,
    pl,
    probegp_featureslist,
    root_mean_squared_error,
    tqdm,
):
    def bootstrap_rmse(df, field_data, config, n_bootstrap=None):
        """Bootstrap analysis with GP - numerically stable version"""
        import warnings

        warnings.filterwarnings("ignore")  # Supprimer les warnings de convergence

        if n_bootstrap is None:
            n_bootstrap = config["BOOTSTRAP_N_SAMPLES"]

        np.random.seed(config["BOOTSTRAP_SEED"])
        unique_soil_ids = df["Soil_ID"].unique().to_list()
        n_soils = len(unique_soil_ids)
        rmse_sm_list = []
        rmse_nd_list = []

        for _i in tqdm(range(n_bootstrap), desc="Bootstrap with GP"):
            _resampled_soil_ids = np.random.choice(
                unique_soil_ids, size=n_soils, replace=True
            )

            _proctor_subset = data_02.filter(
                pl.col("Soil_ID").is_in(_resampled_soil_ids)
            )
            _field_subset = df.filter(pl.col("Soil_ID").is_in(_resampled_soil_ids))

            if len(_proctor_subset) < 10 or len(_field_subset) < 5:
                continue

            # Train probe model (GP)
            _probegp_features_boot = (
                _proctor_subset.select(probegp_featureslist).drop_nulls().to_numpy()
            )
            _probegp_targets_boot = (
                _proctor_subset.select(["WLR"]).drop_nulls().to_numpy()
            )

            _probegp_scaler_boot = RobustScaler()
            _probegp_features_boot_sc = _probegp_scaler_boot.fit_transform(
                _probegp_features_boot
            )

            _probegp_model_boot = GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=1.0,
                    nu=1.5,  # Use reasonable default for bootstrap
                    length_scale_bounds=config["GP_LENGTH_SCALE_BOUNDS"]
                ) + WhiteKernel(noise_level=0.1),
                alpha=0.1,  # Use reasonable default for bootstrap
                n_restarts_optimizer=2,
                normalize_y=True,
                copy_X_train=False,
                random_state=config["BOOTSTRAP_SEED"] + _i,
            )
            _probegp_model_boot.fit(_probegp_features_boot_sc, _probegp_targets_boot)

            # Predict WLR_R1
            _field_with_probe = field_results.filter(
                pl.col("Soil_ID").is_in(_resampled_soil_ids)
            ).rename({"Probe_before_flood": "Probe"})
            _probe_field_features = (
                _field_with_probe.select(probegp_featureslist)
                .drop_nulls()
                .to_numpy()
            )
            _probe_field_features_sc = _probegp_scaler_boot.transform(
                _probe_field_features
            )
            _WLR_R1_boot = _probegp_model_boot.predict(_probe_field_features_sc)

            # Train Sr model (GP)
            _field_other_features = (
                _field_subset.select(["d85", "cu", "Gs"]).drop_nulls().to_numpy()
            )
            field_features_boot = np.column_stack(
                [_WLR_R1_boot, _field_other_features]
            )
            field_targets_boot = (
                _field_subset.select(["SrLR_R2"]).drop_nulls().to_numpy()
            )

            srlr_scaler_boot = RobustScaler()
            field_features_boot_sc = srlr_scaler_boot.fit_transform(
                field_features_boot
            )

            srlr_model_boot = GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=1.0, nu=0.1, length_scale_bounds=(0.1, 20.0)
                )
                + WhiteKernel(noise_level=0.1),
                alpha=0.194,
                n_restarts_optimizer=2,  # Réduit pour vitesse
                normalize_y=True,
                random_state=106638 + _i,
            )
            srlr_model_boot.fit(field_features_boot_sc, field_targets_boot)

            # Predict
            pred_srlr_boot = srlr_model_boot.predict(field_features_boot_sc)
            pred_sr_boot = log_ratio_to_saturation_ratio(
                pred_srlr_boot
            )

            θ_R2_boot = (
                _field_subset.select(["θ_R2"]).drop_nulls().to_numpy().flatten()
            )
            Gs_boot = _field_subset.select(["Gs"]).drop_nulls().to_numpy().flatten()
            ρd_observed_boot = (
                _field_subset.select(["Density_kg/m3_PT"])
                .drop_nulls()
                .to_numpy()
                .flatten()
            )

            ρd_sm_boot = phase_ρd(
                theta=θ_R2_boot, sr=pred_sr_boot.flatten(), rho_w=1000, gs=Gs_boot
            )
            rmse_sm = root_mean_squared_error(ρd_observed_boot, ρd_sm_boot)
            rmse_sm_list.append(rmse_sm)

            # ND bootstrap
            field_nd_boot = field_data.filter(
                pl.col("Soil_ID").is_in(_resampled_soil_ids.tolist())
            )
            field_nd_clean = field_nd_boot.select(
                ["Density_kg/m3_PT", "Density_kg/m3_ND"]
            ).drop_nulls()

            if len(field_nd_clean) > 0:
                rmse_nd = root_mean_squared_error(
                    field_nd_clean["Density_kg/m3_PT"].to_numpy(),
                    field_nd_clean["Density_kg/m3_ND"].to_numpy(),
                )
                rmse_nd_list.append(rmse_nd)

        return np.array(rmse_sm_list), np.array(rmse_nd_list)


    def leave_one_soil_out_cv_rmse(proctor_df, field_df):
        """Leave-One-Soil-Out Cross-Validation (LOSOCV) with GP

        This is NOT traditional LOOCV. Instead, it performs leave-one-soil-out 
        cross-validation where each fold excludes all samples from one soil type,
        preventing data leakage between samples from the same soil.

        Args:
            proctor_df: Proctor data with WLR column (not used - field_df already has everything)
            field_df: Field data with saturation observations (already has WLR_R1, SrLR_R2, etc.)

        Returns:
            Tuple of (overall_rmse, per_soil_rmse_array)
        """
        import polars as pl
        from sklearn.model_selection import GroupKFold
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import RobustScaler
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from tqdm import tqdm

        np.random.seed(config["LOOCV_SEED"])

        # field_df already has WLR_R1, SrLR_R2, θ_R2, and all other needed columns
        field_data_clean = field_df.drop_nulls(subset=["WLR_R1", "SrLR_R2", "θ_R2", "Gs", "Density_kg/m3_PT", "d85", "cu"])

        n_soils = field_data_clean["Soil_ID"].n_unique()
        group_kfold = GroupKFold(n_splits=n_soils)

        # Prepare features and targets for the saturation ratio model
        X_field_full = field_data_clean.select(["WLR_R1", "d85", "cu", "Gs"]).to_numpy()
        y_field_full = field_data_clean.select(["SrLR_R2"]).to_numpy().ravel()
        soil_groups = field_data_clean.select(["Soil_ID"]).to_numpy().ravel()

        all_squared_errors = []
        per_soil_rmse_list = []  # Track RMSE for each soil/fold

        for _train_idx, _test_idx in tqdm(
            group_kfold.split(X_field_full, y_field_full, soil_groups), 
            desc="LOSOCV with GroupKFold", 
            total=n_soils
        ):
            # Split data
            _X_field_train = X_field_full[_train_idx]
            _y_field_train = y_field_full[_train_idx]
            _X_field_test = X_field_full[_test_idx]

            # Scale features
            _scaler_field = RobustScaler()
            _X_field_train_sc = _scaler_field.fit_transform(_X_field_train)
            _X_field_test_sc = _scaler_field.transform(_X_field_test)

            # Train saturation ratio model
            _srmod = GaussianProcessRegressor(
                kernel=Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1),
                n_restarts_optimizer=config.get("N_RESTARTS_OPTIMIZER", 5),
                alpha=0.01,
                normalize_y=True,
                copy_X_train=False,
                random_state=config["LOOCV_SEED"],
            )
            _srmod.fit(_X_field_train_sc, _y_field_train)

            # Predict saturation ratio
            _pred_srlr_test = _srmod.predict(_X_field_test_sc)
            _pred_sr_test = log_ratio_to_saturation_ratio(_pred_srlr_test)

            # Get test field data using the indices
            _test_field_data = field_data_clean[_test_idx]
            _θ_R2_test = _test_field_data.select(["θ_R2"]).to_numpy().flatten()
            _Gs_test = _test_field_data.select(["Gs"]).to_numpy().flatten()
            _ρd_observed_test = _test_field_data.select(["Density_kg/m3_PT"]).to_numpy().flatten()

            # Compute predicted density
            _ρd_sm_test = phase_ρd(theta=_θ_R2_test, sr=_pred_sr_test.flatten(), rho_w=1000, gs=_Gs_test)
            _squared_errors = (_ρd_observed_test - _ρd_sm_test) ** 2
        
            # Track errors for this fold
            fold_rmse = np.sqrt(np.mean(_squared_errors))
            per_soil_rmse_list.append(fold_rmse)
        
            # Also add to overall error pool
            all_squared_errors.extend(_squared_errors)

        if not all_squared_errors:
            return np.nan, np.array([])

        overall_rmse = np.sqrt(np.mean(all_squared_errors))
        per_soil_rmse = np.array(per_soil_rmse_list)

        return overall_rmse, per_soil_rmse
    return bootstrap_rmse, leave_one_soil_out_cv_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""### Run bootstrap analysis""")
    return


@app.cell
def _(bootstrap_rmse, config, field, mo, np, srmod_df):
    with mo.persistent_cache(name="losocv_cache"):
        rmse_sm_bootstrap, rmse_nd_bootstrap = bootstrap_rmse(
            srmod_df, field, config, n_bootstrap=config["BOOTSTRAP_N_SAMPLES"]
        )
        ci_lower_sm = np.percentile(rmse_sm_bootstrap, 2.5)
        ci_upper_sm = np.percentile(rmse_sm_bootstrap, 97.5)
        ci_lower_nd = np.percentile(rmse_nd_bootstrap, 2.5)
        ci_upper_nd = np.percentile(rmse_nd_bootstrap, 97.5)
    return (
        ci_lower_nd,
        ci_lower_sm,
        ci_upper_nd,
        ci_upper_sm,
        rmse_nd_bootstrap,
        rmse_sm_bootstrap,
    )


@app.cell
def _(data_02, field_results, leave_one_soil_out_cv_rmse, mo):
    with mo.persistent_cache(name="bootstrap_cache"):
        rmse_losocv, rmse_losocv_per_soil = leave_one_soil_out_cv_rmse(data_02, field_results)
    return rmse_losocv, rmse_losocv_per_soil


@app.cell
def _(
    ci_lower_nd,
    ci_lower_sm,
    ci_upper_nd,
    ci_upper_sm,
    field,
    log_ratio_to_saturation_ratio,
    np,
    phase_ρd,
    rmse_losocv,
    rmse_nd_bootstrap,
    rmse_sm_bootstrap,
    root_mean_squared_error,
    srmod_df,
):
    ρd_sm_current = phase_ρd(
        theta=srmod_df["θ_R2"],
        sr=log_ratio_to_saturation_ratio(srmod_df["SrLR_R2"]),
        rho_w=1000,
        gs=srmod_df["Gs"],
    )
    rmse_current_sm = root_mean_squared_error(
        srmod_df["Density_kg/m3_PT"], ρd_sm_current
    )

    field_nd_clean = field.select(
        ["Density_kg/m3_PT", "Density_kg/m3_ND"]
    ).drop_nulls()
    rmse_current_nd = root_mean_squared_error(
        field_nd_clean["Density_kg/m3_PT"], field_nd_clean["Density_kg/m3_ND"]
    )

    print(
        f"Bootstrap SM RMSE: {np.mean(rmse_sm_bootstrap):.1f} (95% CI: {ci_lower_sm:.1f}-{ci_upper_sm:.1f})"
    )
    print(
        f"Bootstrap ND RMSE: {np.mean(rmse_nd_bootstrap):.1f} (95% CI: {ci_lower_nd:.1f}-{ci_upper_nd:.1f})"
    )
    print(f"LOOCV SM RMSE: {rmse_losocv:.1f}")
    print(f"Current SM RMSE: {rmse_current_sm:.1f}")
    print(f"Current ND RMSE: {rmse_current_nd:.1f}")
    return (rmse_current_nd,)


@app.cell
def _(rmse_losocv_per_soil):
    rmse_losocv_per_soil
    return


@app.cell
def _(
    ci_lower_nd,
    ci_lower_sm,
    ci_upper_nd,
    ci_upper_sm,
    np,
    pl,
    rmse_current_nd,
    rmse_losocv,
    rmse_losocv_per_soil,
    rmse_sm_bootstrap,
):
    rmse_sm_bootstrap_mean = np.mean(rmse_sm_bootstrap)
    losocv_ci_lower = rmse_losocv_per_soil.min()
    losocv_ci_upper = rmse_losocv_per_soil.max()


    accuracy_summary = pl.DataFrame(
        {
            "Method": [
                "Sherbrooke Method",
                "Sherbrooke Method",
                "Nucleodensimeter",
            ],
            "Validation": [
                f"Bootstrap", #  ({config["BOOTSTRAP_N_SAMPLES"]} samples)
                "Leave-one-soil-out", 
                f"Bootstrap",
            ],
            "RMSE": [rmse_sm_bootstrap_mean, rmse_losocv, rmse_current_nd],
            "CI_lower": [
                ci_lower_sm,
                losocv_ci_lower,
                ci_lower_nd,
            ],
            "CI_upper": [
                ci_upper_sm, 
                losocv_ci_upper,
                ci_upper_nd
            ],
            "Notes": [
                "95% CI",
                "Min-Max range",
                "95% CI"
            ]
        }
    )

    # Round values for display
    accuracy_summary = accuracy_summary.with_columns(
        [
            pl.col("RMSE").round(1),
            pl.col("CI_lower").round(1),
            pl.col("CI_upper").round(1),
        ]
    )
    print(accuracy_summary)
    return (accuracy_summary,)


@app.cell
def _(accuracy_summary, config):
    accuracy_summary.write_csv(config["OUTPUT_ACCURACY"])
    print(f"Accuracy summary table saved to: {config["OUTPUT_ACCURACY"]}")
    return


@app.cell
def _(pred_stats_summary):
    print(pred_stats_summary)
    return


@app.cell
def _(accuracy_summary, pl):
    paper_table =  accuracy_summary.clone().with_columns([
        pl.lit('New Column').alias('status')
    ])
    paper_table
    return


@app.cell
def _(accuracy_summary, pl, pred_stats_summary):
    accuracy_summary_formatted = accuracy_summary.with_columns([
        pl.lit('MG20, MG112').alias('Soil')
    ]).with_columns([
        pl.col('Method').alias('Device'),
        pl.when(
            pl.col('CI_lower').is_not_nan() & 
            pl.col('CI_upper').is_not_nan() &
            pl.col('CI_lower').is_not_null() & 
            pl.col('CI_upper').is_not_null()
        )
        .then(
            pl.format('{} ({}-{})', 
                      pl.col('RMSE').round(1), 
                      pl.col('CI_lower').round(1), 
                      pl.col('CI_upper').round(1))
        )
        .otherwise(pl.col('RMSE').round(1).cast(pl.Utf8))
        .alias('RMSE')
    ]).select(['Device', 'Validation', 'Soil', 'RMSE', 'Notes'])

    pred_stats_formatted = pred_stats_summary.with_columns([
        pl.col('Device'),
        pl.col('Soil type').alias('Soil'),
        pl.col('RMSE').round(1).cast(pl.Utf8),
        pl.lit(' - ').alias('Validation'),
        pl.lit('Fitted on all data').alias('Notes')
    ]).select(['Device', 'Validation', 'Soil', 'RMSE', 'Notes'])

    df_formatted = pl.concat([pred_stats_formatted, accuracy_summary_formatted])

    return (df_formatted,)


@app.cell
def _():
    return


@app.cell
def _(GT, df_formatted):
    great_table = (
        GT(df_formatted)
        .cols_label(
            Device="Device",
            Validation="Validation",
            Soil="Soil",
            RMSE="RMSE",
            Notes="Note"
        )
        .tab_options(
            table_font_size="14px",
            heading_align="left"
        )
    )

    great_table.write_raw_html("images/summary_table.html")
    great_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## Example""")
    return


@app.cell
def _(
    config,
    log_ratio_to_volumetric_water_content,
    np,
    probegp_featuresScaler,
    probegp_model,
):
    np.random.seed(config['EXAMPLE_SEED'])  # random.org

    n_samples = 1000

    probe_examplefeatures_R1 = np.array([[10.0, 0.08, 2.73, 2131]])
    probe_examplefeatures_R2 = np.array([[10.0, 0.08, 2.73, 2342]])

    probe_examplefeatures_R1sc = probegp_featuresScaler.transform(probe_examplefeatures_R1)
    probe_examplefeatures_R2sc = probegp_featuresScaler.transform(probe_examplefeatures_R2)

    probe_WLR1 = probegp_model.sample_y(probe_examplefeatures_R1sc, n_samples=n_samples)
    probe_WLR2 = probegp_model.sample_y(probe_examplefeatures_R2sc, n_samples=n_samples)

    probe_θ1 = log_ratio_to_volumetric_water_content(probe_WLR1)
    probe_θ2 = log_ratio_to_volumetric_water_content(probe_WLR2)
    return n_samples, probe_WLR1, probe_examplefeatures_R1, probe_θ1, probe_θ2


@app.cell
def _(n_samples, np, pd, plt, probe_θ1, probe_θ2, sns):
    probe_θ_df = pd.DataFrame(
        {
            "Value": np.concatenate([probe_θ1[0], probe_θ2[0]]),
            "Probe": ["probe $θ_1$"] * n_samples + ["probe $θ_2$"] * n_samples,
        }
    )

    # Create the histograms
    plt.figure(figsize=(6, 4))
    vwc_ditr = sns.histplot(
        data=probe_θ_df,
        x="Value",
        hue="Probe",
        bins=30,
        kde=False,
        palette=["#777", "#000"],
        edgecolor="white",
        alpha=0.7,
    )
    plt.xlabel("Volumetric water content, $θ$")
    plt.savefig("images/vwc_ditr.png")
    vwc_ditr
    return


@app.cell
def _():
    return


@app.cell
def _(
    log_ratio_to_saturation_ratio,
    n_samples,
    np,
    phase_ρd,
    plt,
    probe_WLR1,
    probe_examplefeatures_R1,
    probe_θ2,
    srlr_model,
    srmod_featuresScaler,
):
    srlr_model_examplefeatures = np.stack(
        [probe_examplefeatures_R1[0]] * n_samples
    )
    srlr_model_examplefeatures[:, -1] = probe_WLR1
    sr_samples = log_ratio_to_saturation_ratio(
        srlr_model.sample_y(
            srmod_featuresScaler.transform(srlr_model_examplefeatures),
            n_samples=n_samples,
        )
    )
    srmod_ρd = phase_ρd(
        theta=probe_θ2, sr=sr_samples, rho_w=1000, gs=probe_examplefeatures_R1[0, 2]
    )
    srmod_ρd_flat = srmod_ρd.flatten()
    density_limit = 1800
    prob_sup = np.sum(srmod_ρd_flat > density_limit) / len(srmod_ρd_flat)

    rho_distr = plt.hist(srmod_ρd_flat, bins=5000, color="#999", edgecolor="white")
    plt.xlim([1000, 2500])
    plt.axvline(density_limit, color="k", linestyle="--")
    plt.title(
        f"Probability to obtain density superior to {density_limit} kg/m³: {round(prob_sup * 100)} %."
    )
    plt.savefig("images/rho_distr.png")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
