import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
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
        "MASTER_SEED": 643126,
        "PROBE_MODEL_SEED": 908493,
        "SR_MODEL_SEED": 71364,
        "BOOTSTRAP_SEED": 381058,
        "LOOCV_SEED": 145018,
        "EXAMPLE_SEED": 592064,

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
        "OPTUNA_N_TRIALS": 50,
        "CV_N_SPLITS": 5,
        "N_RESTARTS_OPTIMIZER": 5,
        "BOOTSTRAP_N_SAMPLES": 300,

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
def _(config):
    # notebook
    import marimo as mo
    from tqdm import tqdm

    # math
    import polars as pl
    import numpy as np
    from scipy.optimize import minimize
    import functools

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    # model
    import optuna
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import (
        mean_squared_error,
        root_mean_squared_error,
        r2_score,
    )
    from sklearn.kernel_ridge import KernelRidge

    np.random.seed(config["MASTER_SEED"])
    return (
        GaussianProcessRegressor,
        KFold,
        KernelRidge,
        Matern,
        RobustScaler,
        WhiteKernel,
        cross_val_score,
        functools,
        mo,
        np,
        optuna,
        pl,
        plt,
        root_mean_squared_error,
        sns,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Utility Functions

    Mathematical transformations and helper functions for bounded variable handling,
    phase relationships, and model evaluation metrics.
    """
    )
    return


@app.cell
def _(config, np):
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
        import optuna
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.model_selection import cross_val_score, KFold, GroupKFold, train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import RobustScaler
        from sklearn.pipeline import Pipeline

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

            # Create pipeline with scaling to improve convergence
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('gp', gp)
            ])

            # Use GroupKFold if soil_ids are provided to prevent data leakage
            if soil_ids is not None:
                cv = GroupKFold(n_splits=min(config["CV_N_SPLITS"], len(np.unique(soil_ids))))
                cv_scores = cross_val_score(
                    pipeline, features, targets, groups=soil_ids, cv=cv, scoring="neg_root_mean_squared_error"
                )
            else:
                cv = KFold(
                    n_splits=config["CV_N_SPLITS"],
                    shuffle=True,
                    random_state=random_seed + 1000
                )
                cv_scores = cross_val_score(
                    pipeline, features, targets, cv=cv, scoring="neg_root_mean_squared_error"
                )

            # Additional overfitting detection: train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, 
                test_size=config.get("OVERFITTING_VAL_SIZE", 0.2), 
                random_state=random_seed + trial.number
            )

            # Create pipeline for validation
            gp_temp = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=alpha, 
                n_restarts_optimizer=config.get("GP_N_RESTARTS_TEMP", 3), 
                copy_X_train=False, 
                normalize_y=config.get("GP_NORMALIZE_Y", True)
            )
            pipeline_temp = Pipeline([
                ('scaler', RobustScaler()),
                ('gp', gp_temp)
            ])

            pipeline_temp.fit(X_train, y_train)

            train_pred = pipeline_temp.predict(X_train)
            val_pred = pipeline_temp.predict(X_val)

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
        saturation_ratio_to_log_ratio,
        split_data_by_soil_ids,
        volumetric_water_content_to_log_ratio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Data

    The `proctor` table includes data from the Proctor tests, `soil` includes the optimized particle-size parameters from the notebook `01_multilevel_rosin.ipynb`, and `field` includes the field data. Some tables are joined, since they share the `Soil_ID` column as key.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Data Validation and Quality Assessment

    Systematic validation of input datasets to ensure data integrity and identify
    potential quality issues before analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    def validate_dataset(df, dataset_name, required_columns=None):
        """
        Validate dataset structure and quality.

        Args:
            df: Polars DataFrame
            dataset_name: Name for reporting
            required_columns: List of required column names

        Returns:
            Validation summary dictionary
        """
        validation = {
            "dataset": dataset_name,
            "shape": df.shape,
            "null_counts": df.null_count().to_dicts()[0],
            "column_types": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "duplicates": df.is_duplicated().sum(),
            "issues": []
        }

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation["issues"].append(f"Missing columns: {missing_cols}")

        null_percentage = {
            col: (count / df.shape[0]) * 100
            for col, count in validation["null_counts"].items()
            if count > 0
        }

        if null_percentage:
            high_null_cols = {col: pct for col, pct in null_percentage.items() if pct > 20}
            if high_null_cols:
                validation["issues"].append(f"High null percentage (>20%): {high_null_cols}")

        return validation

    def print_validation_summary(validation):
        """Print formatted validation summary."""
        print(f"\n{validation['dataset']} Dataset Validation:")
        print(f"  Shape: {validation['shape']}")
        print(f"  Duplicates: {validation['duplicates']}")

        if validation['issues']:
            print("  Issues identified:")
            for issue in validation['issues']:
                print(f"    - {issue}")
        else:
            print("  No critical issues identified")

        null_cols = {k: v for k, v in validation['null_counts'].items() if v > 0}
        if null_cols:
            print(f"  Columns with missing values: {null_cols}")
    return print_validation_summary, validate_dataset


@app.cell
def _(config, pl, print_validation_summary, validate_dataset):
    proctor = pl.read_csv(config["DATA_PROCTOR"])
    soils = pl.read_csv(config["DATA_SOILS"])
    field = pl.read_csv(config["DATA_FIELD"], null_values="NA")

    proctor_validation = validate_dataset(
        proctor, "Proctor",
        required_columns=["Soil_ID", "VolWC_%", "Probe"]
    )
    soils_validation = validate_dataset(
        soils, "Soils",
        required_columns=["Soil_ID", "d85", "cu", "Gs"]
    )
    field_validation = validate_dataset(
        field, "Field",
        required_columns=["Soil_ID", "Probe_before_flood", "Probe_after_1minflood", "Density_kg/m3_PT"]
    )

    for validation in [proctor_validation, soils_validation, field_validation]:
        print_validation_summary(validation)

    data = proctor.join(soils, on="Soil_ID", how="left", coalesce=True)
    data = data.with_columns((pl.col("VolWC_%") / 100).alias("VolWC"))
    return data, field, proctor, soils


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Model Pipeline and Diagnostics

    Advanced modeling framework with sklearn pipelines, prediction intervals,
    and comprehensive model diagnostics for robust geotechnical analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _(np, pl):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin

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

    def create_model_pipeline(model, scaler, feature_engineer=None):
        """
        Create a complete modeling pipeline with preprocessing and feature engineering.

        Args:
            model: Sklearn-compatible model
            scaler: Feature scaler
            feature_engineer: Optional feature engineering transformer

        Returns:
            Sklearn Pipeline object
        """
        steps = []

        if feature_engineer is not None:
            steps.append(('feature_engineer', feature_engineer))

        steps.extend([
            ('scaler', scaler),
            ('model', model)
        ])

        return Pipeline(steps)

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

    def analyze_feature_importance(model, X, feature_names, n_repeats=10, random_state=42):
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
                    model, X, model.predict(X),
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
        SoilFeatureEngineer,
        analyze_feature_importance,
        create_model_pipeline,
        evaluate_model_diagnostics,
        model_robustness_assessment,
        plot_feature_importance,
        plot_residual_diagnostics,
        sensitivity_analysis,
        statistical_model_comparison,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""Let's see how the probe reacts to water content log ratios.""")
    return


@app.cell
def _(data_02, sns):
    sns.scatterplot(data=data_02, x="Probe", y="WLR", hue="Name")
    return


@app.cell(hide_code=True)
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
    SoilFeatureEngineer,
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

    probegp_feature_engineer = SoilFeatureEngineer(include_interactions=True)
    return (
        probegp_feature_engineer,
        probegp_featuresScaler,
        probegp_featureslist,
        probegp_featuressc,
        probegp_targets,
        probegp_test_df,
        probegp_train_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""Optuna is optimizing Gaussian Process hyperparameters (alpha, nu, length_scale) by running multiple trials, evaluating each combination through cross-validation, and progressively converging toward the configuration that minimizes prediction error while controlling for overfitting.""")
    return


@app.cell
def _(
    config,
    optimize_gaussian_process_hyperparameters,
    probegp_featuressc,
    probegp_targets,
    probegp_train_df,
):
    # Extract soil IDs for group-based cross-validation
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
    KFold,
    Matern,
    RobustScaler,
    WhiteKernel,
    config,
    create_model_pipeline,
    cross_val_score,
    evaluate_model_diagnostics,
    log_ratio_to_volumetric_water_content,
    np,
    probe_best_params,
    probegp_feature_engineer,
    probegp_featureslist,
    probegp_test_df,
    probegp_train_df,
):
    np.random.seed(config["PROBE_MODEL_SEED"] + 1000)

    probegp_base_model = GaussianProcessRegressor(
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

    probegp_model = create_model_pipeline(
        model=probegp_base_model,
        scaler=RobustScaler(),
        feature_engineer=probegp_feature_engineer
    )

    probegp_cv = KFold(
        n_splits=config["CV_N_SPLITS"],
        shuffle=True,
        random_state=config["PROBE_MODEL_SEED"] + 3000
    )

    X_train = probegp_train_df.select(probegp_featureslist).to_numpy()
    y_train = probegp_train_df.select(["WLR"]).to_numpy().ravel()

    probegp_rmse_scores = log_ratio_to_volumetric_water_content(
        -cross_val_score(
            probegp_model,
            X_train,
            y_train,
            cv=probegp_cv,
            scoring="neg_root_mean_squared_error",
        )
    )

    probegp_r2_scores = cross_val_score(
        probegp_model,
        X_train,
        y_train,
        cv=probegp_cv,
        scoring="r2",
    )

    probegp_model.fit(X_train, y_train)

    if len(probegp_test_df) > 0:
        X_test = probegp_test_df.select(probegp_featureslist).to_numpy()
        y_test = probegp_test_df.select(["WLR"]).to_numpy().ravel()

        y_pred_test = probegp_model.predict(X_test)

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
    else:
        probe_test_diagnostics = None
        probe_test_diagnostics_vwc = None

    print(f"Probe Model Cross-Validation Results:")
    print(f"  RMSE (VWC) - Min: {np.min(probegp_rmse_scores):.4f}, "
          f"Median: {np.median(probegp_rmse_scores):.4f}, "
          f"Max: {np.max(probegp_rmse_scores):.4f}")
    print(f"  R² - Min: {np.min(probegp_r2_scores):.4f}, "
          f"Median: {np.median(probegp_r2_scores):.4f}, "
          f"Max: {np.max(probegp_r2_scores):.4f}")
    return (probegp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
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
    probegp_featureslist,
    probegp_model,
    probegp_train_df,
):
    if probegp_model is not None:
        X_train_diag = probegp_train_df.select(probegp_featureslist).to_numpy()
        y_train_diag = probegp_train_df.select(["WLR"]).to_numpy().ravel()

        y_pred_train_diag = probegp_model.predict(X_train_diag)

        y_train_vwc_diag = log_ratio_to_volumetric_water_content(y_train_diag)
        y_pred_vwc_diag = log_ratio_to_volumetric_water_content(y_pred_train_diag)

        probe_residual_fig = plot_residual_diagnostics(
            y_train_vwc_diag, y_pred_vwc_diag,
            title="Probe Model Residual Diagnostics (VWC Domain)"
        )
        probe_residual_fig.savefig("images/probe_model_diagnostics.png", dpi=300, bbox_inches='tight')
        probe_residual_fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Advanced Model Analysis and Statistical Testing

    Comprehensive analysis including feature importance, sensitivity analysis,
    model robustness assessment, and statistical significance testing for
    rigorous evaluation of geotechnical modeling performance.
    """
    )
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
    if probegp_model is not None:
        X_probe_importance = probegp_train_df.select(probegp_featureslist).to_numpy()

        probe_importance_data = analyze_feature_importance(
            probegp_model,
            X_probe_importance,
            probegp_featureslist,
            n_repeats=10,
            random_state=config["MASTER_SEED"]
        )

        if probe_importance_data is not None:
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
    if probegp_model is not None:
        X_probe_robust = probegp_train_df.select(probegp_featureslist).to_numpy()
        y_probe_robust = probegp_train_df.select(["WLR"]).to_numpy().ravel()

        probe_robustness = model_robustness_assessment(
            probegp_model,
            X_probe_robust,
            y_probe_robust,
            noise_levels=[0.01, 0.05, 0.1, 0.2]
        )

        if probe_robustness is not None:
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
    if probegp_model is not None and len(probegp_train_df) > 0:
        X_probe_sens = probegp_train_df.select(probegp_featureslist).to_numpy()

        representative_sample = np.median(X_probe_sens, axis=0)

        probe_sensitivity = sensitivity_analysis(
            probegp_model,
            representative_sample,
            probegp_featureslist,
            perturbation_range=0.1,
            n_samples=50
        )

        if probe_sensitivity is not None:
            print("Probe Model Sensitivity Analysis:")
            print("  Feature → Sensitivity | Prediction Range")
            for feature_name in probegp_featureslist:
                sens_data = probe_sensitivity[feature_name]
                print(f"  {feature_name:8} → {sens_data['sensitivity']:10.4f} | {sens_data['prediction_range']:15.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Fit the model and generate predictions.""")
    return


@app.cell
def _(probegp_featuressc, probegp_model, probegp_targets):
    probegp_model.fit(probegp_featuressc, probegp_targets)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We create a new data frame, containing the columns we need, then removing all rows containing at least one null (unmeasured) value.""")
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
    mo.md("""Predict θ_R1 and θ_R2 with one of the models created before.""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Gather the info for modelling.""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Transforming data.""")
    return


@app.cell
def _(field_results):
    field_results
    return


@app.cell
def _(
    RobustScaler,
    SoilFeatureEngineer,
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

    srmod_feature_engineer = SoilFeatureEngineer(include_interactions=True)
    return (
        srmod_df,
        srmod_feature_engineer,
        srmod_features,
        srmod_featuressc,
        srmod_targettr,
        srmod_test_df,
        srmod_train_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Looking for the most appropriate `alpha` with an Optuna search cross validation.""")
    return


@app.cell
def _(
    config,
    optimize_gaussian_process_hyperparameters,
    srmod_featuressc,
    srmod_targettr,
    srmod_train_df,
):
    # Extract soil IDs for group-based cross-validation
    srmod_soil_ids = srmod_train_df["Soil_ID"].to_numpy()

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
    KFold,
    Matern,
    RobustScaler,
    WhiteKernel,
    config,
    create_model_pipeline,
    cross_val_score,
    evaluate_model_diagnostics,
    log_ratio_to_saturation_ratio,
    np,
    sr_best_params,
    srmod_feature_engineer,
    srmod_features,
    srmod_test_df,
    srmod_train_df,
):
    np.random.seed(config["SR_MODEL_SEED"] + 1000)

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

    srlr_model = create_model_pipeline(
        model=srlr_base_model,
        scaler=RobustScaler(),
        feature_engineer=srmod_feature_engineer
    )

    srlr_cv = KFold(
        n_splits=config["CV_N_SPLITS"],
        shuffle=True,
        random_state=config["SR_MODEL_SEED"] + 3000
    )

    X_train_sr = srmod_train_df.select(srmod_features).to_numpy()
    y_train_sr = srmod_train_df.select(["SrLR_R2"]).to_numpy().ravel()

    rmse_scores_sr = log_ratio_to_saturation_ratio(
        -cross_val_score(
            srlr_model,
            X_train_sr,
            y_train_sr,
            cv=srlr_cv,
            scoring="neg_root_mean_squared_error",
        )
    )

    r2_score_srlr = cross_val_score(
        srlr_model,
        X_train_sr,
        y_train_sr,
        cv=srlr_cv,
        scoring="r2",
    )

    srlr_model.fit(X_train_sr, y_train_sr)

    if len(srmod_test_df) > 0:
        X_test_sr = srmod_test_df.select(srmod_features).to_numpy()
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
    else:
        sr_test_diagnostics = None
        sr_test_diagnostics_ratio = None

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
        """
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
    srmod_train_df,
):
    if srlr_model is not None:
        X_sr_importance = srmod_train_df.select(srmod_features).to_numpy()

        sr_importance_data = analyze_feature_importance(
            srlr_model,
            X_sr_importance,
            srmod_features,
            n_repeats=10,
            random_state=config["MASTER_SEED"] + 200
        )

        if sr_importance_data is not None:
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
def _(model_robustness_assessment, srlr_model, srmod_features, srmod_train_df):
    if srlr_model is not None:
        X_sr_robust = srmod_train_df.select(srmod_features).to_numpy()
        y_sr_robust = srmod_train_df.select(["SrLR_R2"]).to_numpy().ravel()

        sr_robustness = model_robustness_assessment(
            srlr_model,
            X_sr_robust,
            y_sr_robust,
            noise_levels=[0.01, 0.05, 0.1, 0.2]
        )

        if sr_robustness is not None:
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
    mo.md("""Compared to nucleodensimeter...""")
    return


@app.cell
def _(abline_ρd, field, pl, sns):
    plot_data_ND = pl.DataFrame(
        {
            "Observed ρd (kg/m³)": field["Density_kg/m3_PT"],
            "Predicted ρd (kg/m³)": field["Density_kg/m3_ND"],
            "Soil type": field["Metatype"],
        }
    )

    abline_ND = [1600, 2400]
    gND = sns.jointplot(
        data=plot_data_ND.to_pandas(),
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
    predND_stats = plot_data_ND.with_columns(
        (
            (pl.col("Observed ρd (kg/m³)") - plot_data_ND["Predicted ρd (kg/m³)"])
            ** 2
        ).alias("squared_errors")
    ).with_columns(pl.Series("Device", ["Nucleodensimeter"] * len(plot_data_ND)))
    return (predND_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Summary statistics for accuracy assessment.""")
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
        .with_columns(pl.Series("Soil type", ["All"] * 2))
        .with_columns(np.sqrt(pl.col("mean_squared_error")).alias("RMSE"))
        .drop("mean_squared_error")
    )
    pred_stats_summary = pred_stats_per_soil.vstack(pred_stats_all)
    pred_stats_summary.write_csv(config["OUTPUT_PRED_STATS"])

    print(f"Prediction statistics summary saved to: {config["OUTPUT_PRED_STATS"]}")
    pred_stats_summary
    return (pred_stats_all,)


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
    KFold,
    KernelRidge,
    cross_val_score,
    functools,
    np,
    optuna,
    probegp_featuressc,
    probegp_targets,
    srmod_featuressc,
    srmod_targettr,
):
    def kernel_ridge_objective(trial, features, target, model_name):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 10.0, log=True)
        kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        cv = KFold(n_splits=5, shuffle=True, random_state=466171)
        score = cross_val_score(
            kr, features, target, cv=cv, scoring="neg_root_mean_squared_error"
        )
        return -np.mean(score)


    # Probe kr model
    probe_kr_study = optuna.create_study(
        study_name="probe KR study", direction="minimize"
    )
    probe_kr_objective_with_data = functools.partial(
        kernel_ridge_objective,
        features=probegp_featuressc,
        target=probegp_targets,
        model_name="probe",
    )
    probe_kr_study.optimize(
        probe_kr_objective_with_data, n_trials=50, show_progress_bar=True
    )

    # Sr kernel ridge model
    sr_kr_study = optuna.create_study(
        study_name="Sr KR study", direction="minimize"
    )
    sr_kr_objective_with_data = functools.partial(
        kernel_ridge_objective,
        features=srmod_featuressc,
        target=srmod_targettr,
        model_name="sr",
    )
    sr_kr_study.optimize(
        sr_kr_objective_with_data, n_trials=50, show_progress_bar=True
    )
    return


@app.cell(hide_code=True)
def _(
    GaussianProcessRegressor,
    Matern,
    RobustScaler,
    WhiteKernel,
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
            pred_sr_boot = log_ratio_to_saturation_ratio(pred_srlr_boot)

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

        This approach is more realistic for geotechnical applications where
        the goal is to predict performance on entirely new soil types.

        Uses GroupKFold with n_splits = number of unique soils for efficiency.
        """
        import warnings
        from sklearn.model_selection import GroupKFold
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import RobustScaler

        warnings.filterwarnings("ignore")

        # Prepare combined dataset for GroupKFold
        field_with_proctor_data = field_df.join(
            proctor_df.select(["Soil_ID"]).unique(),
            on="Soil_ID",
            how="inner"
        )

        if len(field_with_proctor_data) == 0:
            return np.nan

        unique_soil_ids = field_with_proctor_data["Soil_ID"].unique().to_list()
        n_soils = len(unique_soil_ids)

        if n_soils < 2:
            return np.nan

        # Use GroupKFold instead of manual loop
        group_kfold = GroupKFold(n_splits=n_soils)

        # Prepare features and targets for the saturation ratio model
        X_field_full = field_with_proctor_data.select(["WLR_R1", "d85", "cu", "Gs"]).to_numpy()
        y_field_full = field_with_proctor_data.select(["SrLR_R2"]).to_numpy().ravel()
        soil_groups = field_with_proctor_data.select(["Soil_ID"]).to_numpy().ravel()

        all_squared_errors = []

        for _train_idx, _test_idx in tqdm(
            group_kfold.split(X_field_full, y_field_full, soil_groups), 
            desc="LOSOCV with GroupKFold", 
            total=n_soils
        ):
            # Get training and test soil IDs
            _test_soil_ids = np.unique(soil_groups[_test_idx])
            _train_soil_ids = np.unique(soil_groups[_train_idx])

            # Filter proctor data for training soils
            _train_proctor = proctor_df.filter(pl.col("Soil_ID").is_in(_train_soil_ids.tolist()))
            if len(_train_proctor) < 10:
                continue

            # Train probe model
            _probegp_features_train = _train_proctor.select(["d85", "cu", "Gs", "Probe"]).drop_nulls().to_numpy()
            _probegp_targets_train = _train_proctor.select(["WLR"]).drop_nulls().to_numpy().ravel()

            if len(_probegp_features_train) < 10:
                continue

            _probegp_scaler_cv = RobustScaler()
            _probegp_features_train_sc = _probegp_scaler_cv.fit_transform(_probegp_features_train)

            _probegp_model_cv = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 20.0)) + WhiteKernel(noise_level=0.1),
                alpha=0.03141,
                n_restarts_optimizer=2,
                normalize_y=True,
                copy_X_train=False,
                random_state=42
            )
            _probegp_model_cv.fit(_probegp_features_train_sc, _probegp_targets_train)

            # Get field data for test soils and predict WLR_R1
            _test_field_data = field_df.filter(pl.col("Soil_ID").is_in(_test_soil_ids.tolist()))
            _test_probe_features = _test_field_data.select(["d85", "cu", "Gs"]).to_numpy()
            _test_probe_data = np.column_stack([
                _test_probe_features,
                _test_field_data.select(["Probe_before_flood"]).to_numpy().ravel()
            ])

            _test_probe_data_sc = _probegp_scaler_cv.transform(_test_probe_data)
            _WLR_R1_test = _probegp_model_cv.predict(_test_probe_data_sc)

            # Train saturation ratio model
            _train_field = field_df.filter(pl.col("Soil_ID").is_in(_train_soil_ids.tolist()))
            _srlr_features_train = _train_field.select(["WLR_R1", "d85", "cu", "Gs"]).drop_nulls().to_numpy()
            _srlr_targets_train = _train_field.select(["SrLR_R2"]).drop_nulls().to_numpy().ravel()

            if len(_srlr_features_train) < 5:
                continue

            _srlr_scaler_cv = RobustScaler()
            _srlr_features_train_sc = _srlr_scaler_cv.fit_transform(_srlr_features_train)

            _srlr_model_cv = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=0.1, length_scale_bounds=(0.1, 20.0)) + WhiteKernel(noise_level=0.1),
                alpha=0.194,
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=42,
            )
            _srlr_model_cv.fit(_srlr_features_train_sc, _srlr_targets_train)

            # Predict on test data
            _srlr_features_test = np.column_stack([_WLR_R1_test, _test_probe_features])
            _srlr_features_test_sc = _srlr_scaler_cv.transform(_srlr_features_test)
            _pred_srlr_test = _srlr_model_cv.predict(_srlr_features_test_sc)
            _pred_sr_test = log_ratio_to_saturation_ratio(_pred_srlr_test)

            # Calculate errors
            _θ_R2_test = _test_field_data.select(["θ_R2"]).drop_nulls().to_numpy().flatten()
            _Gs_test = _test_field_data.select(["Gs"]).drop_nulls().to_numpy().flatten()
            _ρd_observed_test = _test_field_data.select(["Density_kg/m3_PT"]).drop_nulls().to_numpy().flatten()

            _ρd_sm_test = phase_ρd(theta=_θ_R2_test, sr=_pred_sr_test.flatten(), rho_w=1000, gs=_Gs_test)
            _squared_errors = (_ρd_observed_test - _ρd_sm_test) ** 2
            all_squared_errors.extend(_squared_errors)

        return np.sqrt(np.mean(all_squared_errors)) if all_squared_errors else np.nan

        unique_soil_ids = field_df["Soil_ID"].unique().to_list()
        all_squared_errors = []

        for i, test_soil_id in tqdm(enumerate(unique_soil_ids), desc="LOOCV with GP"):
            train_proctor = proctor_df.filter(pl.col("Soil_ID") != test_soil_id)
            train_field = field_df.filter(pl.col("Soil_ID") != test_soil_id)
            test_field = field_df.filter(pl.col("Soil_ID") == test_soil_id)

            if (
                len(train_proctor) < 10
                or len(train_field) < 5
                or len(test_field) == 0
            ):
                continue

            # Train probe model
            probegp_features_train = (
                train_proctor.select(probegp_featureslist).drop_nulls().to_numpy()
            )
            probegp_targets_train = (
                train_proctor.select(["WLR"]).drop_nulls().to_numpy()
            )

            probegp_scaler_cv = RobustScaler()
            probegp_features_train_sc = probegp_scaler_cv.fit_transform(
                probegp_features_train
            )

            probegp_model_cv = GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 20.0)
                )
                + WhiteKernel(noise_level=0.1),
                alpha=0.03141,
                n_restarts_optimizer=2,
                normalize_y=True,
                copy_X_train=False,
                random_state=42 + i,
            )
            probegp_model_cv.fit(probegp_features_train_sc, probegp_targets_train)

            # Predict WLR_R1
            test_with_probe = field_results.filter(
                pl.col("Soil_ID") == test_soil_id
            ).rename({"Probe_before_flood": "Probe"})
            probe_test_features = (
                test_with_probe.select(probegp_featureslist)
                .drop_nulls()
                .to_numpy()
            )
            probe_test_features_sc = probegp_scaler_cv.transform(
                probe_test_features
            )
            WLR_R1_test = probegp_model_cv.predict(probe_test_features_sc)

            # Train Sr model
            srlr_features_train = (
                train_field.select(["WLR_R1", "d85", "cu", "Gs"])
                .drop_nulls()
                .to_numpy()
            )
            srlr_targets_train = (
                train_field.select(["SrLR_R2"]).drop_nulls().to_numpy()
            )

            srlr_scaler_cv = RobustScaler()
            srlr_features_train_sc = srlr_scaler_cv.fit_transform(
                srlr_features_train
            )

            srlr_model_cv = GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=1.0, nu=0.1, length_scale_bounds=(0.1, 20.0)
                )
                + WhiteKernel(noise_level=0.1),
                alpha=0.194,
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=42,
            )
            srlr_model_cv.fit(srlr_features_train_sc, srlr_targets_train)

            # Build test features and predict
            test_other_features = (
                test_field.select(["d85", "cu", "Gs"]).drop_nulls().to_numpy()
            )
            srlr_features_test = np.column_stack(
                [WLR_R1_test, test_other_features]
            )
            srlr_features_test_sc = srlr_scaler_cv.transform(srlr_features_test)

            pred_srlr_test = srlr_model_cv.predict(srlr_features_test_sc)
            pred_sr_test = log_ratio_to_saturation_ratio(pred_srlr_test)

            θ_R2_test = (
                test_field.select(["θ_R2"]).drop_nulls().to_numpy().flatten()
            )
            Gs_test = test_field.select(["Gs"]).drop_nulls().to_numpy().flatten()
            ρd_observed_test = (
                test_field.select(["Density_kg/m3_PT"])
                .drop_nulls()
                .to_numpy()
                .flatten()
            )

            ρd_sm_test = phase_ρd(
                theta=θ_R2_test, sr=pred_sr_test.flatten(), rho_w=1000, gs=Gs_test
            )
            squared_errors = (ρd_observed_test - ρd_sm_test) ** 2
            all_squared_errors.extend(squared_errors)

        return (
            np.sqrt(np.mean(all_squared_errors)) if all_squared_errors else np.nan
        )
    return bootstrap_rmse, leave_one_soil_out_cv_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run bootstrap analysis""")
    return


@app.cell
def _(
    bootstrap_rmse,
    config,
    data_02,
    field,
    field_results,
    leave_one_soil_out_cv_rmse,
    np,
    srmod_df,
):
    rmse_sm_bootstrap, rmse_nd_bootstrap = bootstrap_rmse(
        srmod_df, field, config, n_bootstrap=config["BOOTSTRAP_N_SAMPLES"]
    )
    ci_lower_sm = np.percentile(rmse_sm_bootstrap, 2.5)
    ci_upper_sm = np.percentile(rmse_sm_bootstrap, 97.5)
    ci_lower_nd = np.percentile(rmse_nd_bootstrap, 2.5)
    ci_upper_nd = np.percentile(rmse_nd_bootstrap, 97.5)

    rmse_losocv = leave_one_soil_out_cv_rmse(data_02, field_results)
    return (
        ci_lower_nd,
        ci_lower_sm,
        ci_upper_nd,
        ci_upper_sm,
        rmse_losocv,
        rmse_nd_bootstrap,
        rmse_sm_bootstrap,
    )


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
def _(
    ci_lower_nd,
    ci_lower_sm,
    ci_upper_nd,
    ci_upper_sm,
    np,
    pl,
    pred_stats_all,
    rmse_current_nd,
    rmse_losocv,
):
    # Create summary table

    rmse_all_cv = (
        pred_stats_all.filter(pl.col("Device") == "Sherbrooke Method")
        .select("RMSE")
        .to_numpy()
    )[0][0]

    accuracy_summary = pl.DataFrame(
        {
            "Method": [
                "Sherbrooke Method",
                "Sherbrooke Method",
                "Nucleodensimeter",
            ],
            "Cross-validation": [
                "Random 5-fold",
                "Leave-one-soil-out",
                "Same test points",
            ],
            "RMSE": [rmse_all_cv, rmse_losocv, rmse_current_nd],
            "CI_lower": [
                ci_lower_sm,
                np.nan,
                ci_lower_nd,
            ],  # LOOCV doesn't have bootstrap CI
            "CI_upper": [ci_upper_sm, np.nan, ci_upper_nd],
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

    print("Accuracy Summary Table:")
    print(accuracy_summary)
    return (accuracy_summary,)


@app.cell
def _(accuracy_summary, config):
    accuracy_summary.write_csv(config["OUTPUT_ACCURACY"])
    print(f"Accuracy summary table saved to: {config["OUTPUT_ACCURACY"]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Statistical Model Comparison and Final Assessment

    Rigorous statistical comparison between the Sherbrooke Method and nucleodensimeter
    using paired statistical tests to determine significant performance differences.
    """
    )
    return


@app.cell
def _(
    config,
    field,
    field_results,
    log_ratio_to_saturation_ratio,
    phase_ρd,
    srlr_model,
    srmod_features,
    statistical_model_comparison,
):

    field_comparison = field.select([
        "Soil_ID", "Density_kg/m3_PT", "Density_kg/m3_ND",
        "Metatype"
    ]).drop_nulls()

    # For statistical comparison, we need existing field results with predictions
    # This assumes field_results DataFrame exists with all necessary columns

    # Use previously computed field_results if available
    field_for_comparison = field_results.select([
        "Soil_ID", "WLR_R1", "d85", "cu", "Gs", "θ_R2", "Density_kg/m3_PT"
    ]).drop_nulls()

    # Get corresponding nucleodensimeter data for the same samples
    nd_comparison = field.select([
        "Soil_ID", "Density_kg/m3_ND"
    ]).drop_nulls()

    # Join to ensure we have the same samples for both methods
    comparison_data = field_for_comparison.join(
        nd_comparison, 
        on="Soil_ID", 
        how="inner"
    )

    X_field_comparison = comparison_data.select(srmod_features).to_numpy()
    y_true_density = comparison_data.select(["Density_kg/m3_PT"]).to_numpy().ravel()

    srlr_pred_comparison = srlr_model.predict(X_field_comparison)
    sr_pred_comparison = log_ratio_to_saturation_ratio(srlr_pred_comparison)

    theta_R2_comparison = comparison_data.select(["θ_R2"]).to_numpy().ravel()
    gs_comparison = comparison_data.select(["Gs"]).to_numpy().ravel()

    sherbrooke_pred = phase_ρd(
        theta=theta_R2_comparison,
        sr=sr_pred_comparison,
        rho_w=config["WATER_DENSITY"],
        gs=gs_comparison
    )

    nd_pred = comparison_data.select(["Density_kg/m3_ND"]).to_numpy().ravel()


    comparison_results = statistical_model_comparison(
        y_true_density,
        sherbrooke_pred,
        nd_pred,
        model_names=["Sherbrooke Method", "Nucleodensimeter"]
    )

    print("Statistical Comparison: Sherbrooke Method vs. Nucleodensimeter")
    print("=" * 65)
    print(f"Sample size: {comparison_results['sample_size']}")
    print(f"Sherbrooke Method MAE: {comparison_results['mean_error_1']:.2f} kg/m³")
    print(f"Nucleodensimeter MAE: {comparison_results['mean_error_2']:.2f} kg/m³")
    print(f"Mean error difference: {comparison_results['error_difference']:.2f} kg/m³")
    print()
    print("Statistical Tests:")
    print(f"  Paired t-test:")
    print(f"    t-statistic: {comparison_results['t_statistic']:.4f}")
    print(f"    p-value: {comparison_results['t_pvalue']:.6f}")
    print(f"  Wilcoxon signed-rank test:")
    print(f"    statistic: {comparison_results['wilcoxon_statistic']:.1f}")
    print(f"    p-value: {comparison_results['wilcoxon_pvalue']:.6f}")
    print(f"  Effect size (Cohen's d): {comparison_results['cohens_d']:.4f}")
    print()

    significance_level = 0.05
    is_significant_t = comparison_results['t_pvalue'] < significance_level
    is_significant_w = comparison_results['wilcoxon_pvalue'] < significance_level

    print("Interpretation:")
    if is_significant_t and is_significant_w:
        print("  Both tests indicate statistically significant difference (p < 0.05)")
    elif is_significant_t or is_significant_w:
        print("  One test indicates significant difference - results inconclusive")
    else:
        print("  No statistically significant difference detected (p ≥ 0.05)")

    if abs(comparison_results['cohens_d']) < 0.2:
        effect_size_desc = "negligible"
    elif abs(comparison_results['cohens_d']) < 0.5:
        effect_size_desc = "small"
    elif abs(comparison_results['cohens_d']) < 0.8:
        effect_size_desc = "medium"
    else:
        effect_size_desc = "large"

    print(f"  Effect size is {effect_size_desc} (|d| = {abs(comparison_results['cohens_d']):.3f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## Example""")
    return


@app.cell
def _(
    config,
    log_ratio_to_saturation_ratio,
    log_ratio_to_volumetric_water_content,
    np,
    probegp_model,
    srlr_model,
):
    np.random.seed(config['EXAMPLE_SEED'])  # random.org

    n_samples = 1000

    probe_examplefeatures_R1 = np.array([[10.0, 0.08, 2.73, 2131]])
    probe_examplefeatures_R2 = np.array([[10.0, 0.08, 2.73, 2342]])

    # Apply all pipeline transformations except the final model step
    probe_features_R1_transformed = probe_examplefeatures_R1.copy()
    probe_features_R2_transformed = probe_examplefeatures_R2.copy()

    # Apply each transformation step in the pipeline except the model
    for name, transformer in probegp_model.named_steps.items():
        if name != 'model':
            probe_features_R1_transformed = transformer.transform(probe_features_R1_transformed)
            probe_features_R2_transformed = transformer.transform(probe_features_R2_transformed)

    probe_WLR1 = probegp_model.named_steps['model'].sample_y(
        probe_features_R1_transformed,
        n_samples=n_samples,
    )[0]
    probe_WLR2 = probegp_model.named_steps['model'].sample_y(
        probe_features_R2_transformed,
        n_samples=n_samples,
    )[0]

    probe_θ1 = log_ratio_to_volumetric_water_content(probe_WLR1)
    probe_θ2 = log_ratio_to_volumetric_water_content(probe_WLR2)
    srlr_model_examplefeatures = np.stack(
        [probe_examplefeatures_R1[0]] * n_samples
    )
    srlr_model_examplefeatures[:, -1] = probe_WLR1

    # Apply all pipeline transformations except the final model step for srlr_model
    srlr_features_transformed = srlr_model_examplefeatures.copy()
    for name, transformer in srlr_model.named_steps.items():
        if name != 'model':
            srlr_features_transformed = transformer.transform(srlr_features_transformed)

    sr_samples = log_ratio_to_saturation_ratio(
        srlr_model.named_steps['model'].sample_y(
            srlr_features_transformed,
            n_samples=n_samples,
        )
    )
    return n_samples, probe_examplefeatures_R1, probe_θ1, probe_θ2, sr_samples


@app.cell
def _(phase_ρd, probe_examplefeatures_R1, probe_θ2, sr_samples):
    srmod_ρd = phase_ρd(
        theta=probe_θ2, sr=sr_samples, rho_w=1000, gs=probe_examplefeatures_R1[0, 2]
    )
    return (srmod_ρd,)


@app.cell
def _(n_samples, np, pl, plt, probe_θ1, probe_θ2, sns):
    probe_θ2
    probe_θ_df = pl.DataFrame(
        {
            "Value": np.concatenate([probe_θ1, probe_θ2]),
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
def _(np, plt, srmod_ρd):
    srmod_ρd_flat = srmod_ρd.flatten()
    density_limit = 1800
    prob_sup = np.sum(srmod_ρd_flat > density_limit) / len(srmod_ρd_flat)

    rho_distr = plt.hist(srmod_ρd_flat, bins=600, color="#999", edgecolor="#333")
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
