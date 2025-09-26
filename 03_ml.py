import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Sherbrooke Method Modelling

    ## Principles

    **Calibration of the Probe**. Starting from particle-size parameters $g_1$ and $g_2$, as well as the specific gravity $Gs$, we predict an *exponential association* curve comprising three parameters, $y_{min}$, $y_{max}$, and the $slope$ which best predicts the soil water content $θ$ as a function of the probe measurement $M$, based on Proctor tests.

    **Sherbrooke Method**. The particle-size parameters $g_1$ and $g_2$, the specific gravity $Gs$, as well as the water content measured before flooding $θ_{R1}$, allow estimating the degree of saturation obtained after one minute of wetting $S_{opt}$. With $S_{opt}$ and the water content measured after the flood $θ_{R2}$, we can obtain the dry density $ρ_{d}$ with $ρ_w$ and the $Gs$.

    ## Packages

    We will need the Polars module for importing and manipulating tabular data, Numpy for matrix calculations, Scikit-learn for machine learning and, for graphics, Matplotlib and Let's-plot.
    """
    )
    return


@app.cell
def _():
    # notebook
    import marimo as mo
    from tqdm import tqdm

    # math
    import polars as pl
    import numpy as np

    np.random.seed(554390)  # random.org
    from scipy.optimize import minimize
    import functools

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    # model
    import optuna
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import (
        mean_squared_error,
        root_mean_squared_error,
        r2_score,
    )
    from sklearn.kernel_ridge import KernelRidge

    np.random.seed(7235)
    return (
        GaussianProcessRegressor,
        KFold,
        KernelRidge,
        Matern,
        StandardScaler,
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


@app.cell
def _(np):
    def mean_variance_index(scores, weight=0.7):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        normalized_mean = (mean_score - np.min(scores)) / (
            np.max(scores) - np.min(scores)
        )
        normalized_std = 1 - (std_score / np.max(scores))
        combined_score = weight * normalized_mean + (1 - weight) * normalized_std
        return combined_score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Data

    The `proctor` table includes data from the Proctor tests, `soil` includes the optimized particle-size parameters from the notebook `01_multilevel_rosin.ipynb`, and `field` includes the field data. Some tables are joined, since they share the `Soil_ID` column as key.
    """
    )
    return


@app.cell
def _(pl):
    proctor = pl.read_csv("data/r_proctor.csv")
    soils = pl.read_csv("data/soils_rosin.csv")
    field = pl.read_csv("data/r_field.csv", null_values="NA")
    data = proctor.join(soils, on="Soil_ID", how="left", coalesce=True)
    data = data.with_columns((pl.col("VolWC_%") / 100).alias("VolWC"))
    data
    return data, field, proctor, soils


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""Overview of data quantity.""")
    return


@app.cell
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
def _(data, np, pl):
    def VWC_to_WLR(x):
        # transform volumetric water content to water log ratio
        return np.log(x / (1 - x))


    def WLR_to_VWC(x):
        # transform water log ratio to volumetric water content
        return np.exp(x) / (1 + np.exp(x))


    data_02 = data.with_columns((VWC_to_WLR(pl.col("VolWC"))).alias("WLR"))
    return WLR_to_VWC, data_02


@app.cell
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
def _(StandardScaler, data_02):
    # clean data
    probegp_featureslist = ["d85", "cu", "Gs", "Probe"]
    probegp_targetslist = ["WLR"]
    probegp_df = data_02[probegp_featureslist + probegp_targetslist].drop_nulls()

    # features
    probegp_features = probegp_df.select(probegp_featureslist).to_numpy()
    probegp_featuresScaler = StandardScaler()
    probegp_featuressc = probegp_featuresScaler.fit_transform(probegp_features)

    # targets
    probegp_targets = probegp_df.select(probegp_targetslist).to_numpy()
    return (
        probegp_featuresScaler,
        probegp_featureslist,
        probegp_featuressc,
        probegp_targets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""A Optuna serach can help finding optimal hyperparameters...""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```
    np.random.seed(602886)  # random.org

    def probe_objective(trial, features, target):
        alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
        nu = trial.suggest_float("nu", 0.1, 1.5)
        kernel = Matern(
            length_scale=1.0, nu=nu, length_scale_bounds=(0.01, 10.0)
        ) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, n_restarts_optimizer=5, copy_X_train=False
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=466171)  # random.org
        score = cross_val_score(
            gp, features, target, cv=cv, scoring="neg_root_mean_squared_error"
        )
        index = mean_variance_index(score, weight=0.7)
        return index


    probe_objective_with_data = functools.partial(
        probe_objective, features=probegp_featuressc, target=probegp_targets
    )
    probe_study = optuna.create_study(
        study_name="probe GP study", direction="maximize"
    )
    probe_study.optimize(
        probe_objective_with_data, n_trials=50, show_progress_bar=True
    )
    probe_study.best_params
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The best parameters were found at iteration 24 with values of {'alpha': 0.03140764053242634, 'nu': 1.4969865518170358}."""
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    KFold,
    Matern,
    WLR_to_VWC,
    WhiteKernel,
    cross_val_score,
    np,
    probegp_featuressc,
    probegp_targets,
):
    # model
    np.random.seed(774666)  # random.org
    probegp_model = GaussianProcessRegressor(
        kernel=Matern(length_scale=1.0, nu=1.497, length_scale_bounds=(0.01, 20.0))
        + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=10,
        alpha=0.03141,
        optimizer="fmin_l_bfgs_b",
        normalize_y=True,
        copy_X_train=False,
        random_state=909193,  # random.org
    )
    probegp_cv = KFold(n_splits=5, shuffle=True, random_state=107838)  # random.org
    probegp_rmse_scores = WLR_to_VWC(
        -cross_val_score(
            probegp_model,
            probegp_featuressc,
            probegp_targets,
            cv=probegp_cv,
            scoring="neg_root_mean_squared_error",
        )
    )

    probegp_r2_scores = cross_val_score(
        probegp_model,
        probegp_featuressc,
        probegp_targets,
        cv=probegp_cv,
        scoring="r2",
    )

    print(f"Min RMSE: {np.min(probegp_rmse_scores)}")
    print(f"Median RMSE: {np.median(probegp_rmse_scores)}")
    print(f"Max RMSE: {np.max(probegp_rmse_scores)}")

    print(f"Min R2: {np.min(probegp_r2_scores)}")
    print(f"Median R2: {np.median(probegp_r2_scores)}")
    print(f"Max R2: {np.max(probegp_r2_scores)}")
    return (probegp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Fit the model.""")
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
def _(
    WLR_to_VWC,
    data,
    pl,
    probegp_featuressc,
    probegp_model,
    probegp_targets,
    sns,
):
    probegp_model_targets_pred = WLR_to_VWC(
        probegp_model.predict(probegp_featuressc)
    )
    probegp_model_targets_obs = WLR_to_VWC(probegp_targets.flatten())
    plot_data = pl.DataFrame(
        {
            "Observed θ in Proctor testing": probegp_model_targets_obs,
            "Predicted θ with GP modelling": probegp_model_targets_pred,
            "Soil type": data["Metatype"],
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
    WLR_to_VWC,
    data_02,
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
        probegp_model_predtargets = WLR_to_VWC(
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
def _():
    def phase_ρd(θ, Sr, ρw, Gs):
        ρd = (1 - θ / Sr) * Gs * ρw
        return ρd


    def phase_Sr(θ, ρd, ρw, Gs):
        Sr = θ / (1 - ρd / (Gs * ρw))
        return Sr


    def porosity(ρd, ρw, Gs):
        return 1 - ρd / (Gs * ρw)
    return phase_Sr, phase_ρd, porosity


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""Data from table `field`, containing results from field experiments, will be used. We'll need $d_{85}$ and $Cu$, which are joined to the `field` table."""
    )
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
def _(np):
    def sr_to_srlr(sr):
        return np.log(sr / (1 - sr))


    def srlr_to_sr(srlr, maxsr=0.99):
        return np.minimum(maxsr, np.exp(srlr) / (1 + np.exp(srlr)))
    return sr_to_srlr, srlr_to_sr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """We create a new data frame, containing the columns we need, then removing all rows containing at least one null (unmeasured) value."""
    )
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
    WLR_to_VWC,
    field_full,
    phase_Sr,
    pl,
    porosity,
    sr_to_srlr,
):
    θ_R1 = WLR_to_VWC(WLR_R1)
    θ_R2 = WLR_to_VWC(WLR_R2)

    field_results = (
        field_full
        # insert porosity
        .with_columns(
            porosity(
                ρd=pl.col("Density_kg/m3_PT"), ρw=1000, Gs=pl.col("Gs")
            ).alias("porosity")
        )
        # insert other variables
        .with_columns(
            [
                pl.Series(WLR_R1).alias("WLR_R1"),
                pl.Series(WLR_R2).alias("WLR_R2"),
                pl.Series(θ_R1).alias("θ_R1"),
                pl.Series(θ_R2).alias("θ_R2"),
                pl.lit(1000.0).alias("ρw"),
            ]
        )
        # make sure volumetric water content does not exceed porosity
        .with_columns(
            [
                pl.when(pl.col("θ_R1") < pl.col("porosity"))
                .then(pl.col("θ_R1"))
                .otherwise(pl.col("porosity") * 0.99),
                pl.when(pl.col("θ_R2") < pl.col("porosity"))
                .then(pl.col("θ_R2"))
                .otherwise(pl.col("porosity") * 0.99),
            ]
        )
        # compute Sr
        .with_columns(
            phase_Sr(
                θ=pl.col("θ_R2"),
                ρd=pl.col("Density_kg/m3_PT"),
                ρw=pl.col("ρw"),
                Gs=pl.col("Gs"),
            ).alias("Sr_R2")
        )
        .with_columns(sr_to_srlr(pl.col("Sr_R2")).alias("SrLR_R2"))
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
def _(StandardScaler, field_results, pl):
    # clean data
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
    )  # Density_kg/m3_PT to keep track of it

    # features
    srmod_featurestr = srmod_df[srmod_features].to_numpy()
    srmod_featuresScaler = StandardScaler()
    srmod_featuresScaler.fit_transform(srmod_featurestr)
    srmod_featuresmean = srmod_featurestr.mean(axis=0)
    srmod_featuresstd = srmod_featurestr.std(axis=0)
    srmod_featuressc = (srmod_featurestr - srmod_featuresmean) / srmod_featuresstd

    # targets
    srmod_targettr = srmod_df[srmod_target].to_numpy()
    return srmod_df, srmod_featuresScaler, srmod_featuressc, srmod_targettr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """Looking for the most appropriate `alpha` with an Optuna search cross validation."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```python
    np.random.seed(534474)  # random.org


    def sr_objective(trial, features, target):
        alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
        nu = trial.suggest_float("nu", 0.1, 1.5)
        kernel = Matern(
            length_scale=1.0, nu=nu, length_scale_bounds=(0.01, 10.0)
        ) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=5,
            copy_X_train=False,
            random_state=470943,
        )  # random.org
        srlr_cv = KFold(
            n_splits=5, shuffle=True, random_state=269428
        )  # random.org
        score = cross_val_score(
            gp, features, target, cv=srlr_cv, scoring="neg_root_mean_squared_error"
        )
        index = mean_variance_index(score, weight=0.7)
        return index


    sr_objective_with_data = functools.partial(
        sr_objective, features=srmod_featuressc, target=srmod_targettr
    )
    sr_study = optuna.create_study(study_name="Sr GP study", direction="maximize")
    sr_study.optimize(sr_objective_with_data, n_trials=50, show_progress_bar=True)
    sr_study.best_params
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """We then model Sr with a Gaussian process with the best parameters, obtained at iteration 31: `{'alpha': 0.19395414063620006, 'nu': 0.10087920710585421}`."""
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    KFold,
    Matern,
    WhiteKernel,
    cross_val_score,
    np,
    srlr_to_sr,
    srmod_featuressc,
    srmod_targettr,
):
    np.random.seed(136050)  # random.org
    srlr_kernel = Matern(
        length_scale=1.0, nu=0.1009, length_scale_bounds=(0.01, 20.0)
    ) + WhiteKernel(noise_level=0.1)
    srlr_model = GaussianProcessRegressor(
        kernel=srlr_kernel,
        n_restarts_optimizer=10,
        alpha=0.1940,
        normalize_y=True,
        random_state=589919,
    )  # random.org
    srlr_cv = KFold(n_splits=5, shuffle=True, random_state=977455)  # random.org
    rmse_scores_sr = srlr_to_sr(
        -cross_val_score(
            srlr_model,
            srmod_featuressc,
            srmod_targettr,
            cv=srlr_cv,
            scoring="neg_root_mean_squared_error",
        )
    )
    r2_score_srlr = cross_val_score(
        srlr_model,
        srmod_featuressc,
        srmod_targettr,
        cv=srlr_cv,
        scoring="r2",
    )

    print(f"Min RMSE : {np.min(rmse_scores_sr)}")
    print(f"Median RMSE : {np.median(rmse_scores_sr)}")
    print(f"Max RMSE : {np.max(rmse_scores_sr)}")

    print(f"Min R2 : {np.min(r2_score_srlr)}")
    print(f"Median R2 : {np.median(r2_score_srlr)}")
    print(f"Max R2 : {np.max(r2_score_srlr)}")
    return (srlr_model,)


@app.cell
def _(
    root_mean_squared_error,
    srlr_model,
    srlr_to_sr,
    srmod_featuressc,
    srmod_targettr,
):
    srlr_model.fit(srmod_featuressc, srmod_targettr)
    srlr_pred = srlr_model.predict(srmod_featuressc)
    print(f"RMSE for Sr: {root_mean_squared_error(srmod_targettr, srlr_pred)}")
    sr_pred = srlr_to_sr(srlr_pred)
    return (sr_pred,)


@app.cell
def _(pl, sns, sr_pred, srlr_to_sr, srmod_df):
    plot_data_sr = pl.DataFrame(
        {
            "Observed Sr in feild testing": srlr_to_sr(
                srmod_df["SrLR_R2"].to_numpy()
            ),
            "Predicted Sr with GP modelling": sr_pred,
            "Soil type": srmod_df["Metatype"],
        }
    )
    abline_sr = [0.4, 1.0]
    gsr = sns.jointplot(
        data=plot_data_sr,
        x="Observed Sr in feild testing",
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
    mo.md(
        rf"""Once Sr is modelled, predictions can be expressed in terms of $ρ_d$."""
    )
    return


@app.cell
def _(phase_ρd, pl, sns, sr_pred, srmod_df):
    plot_data_ρd = pl.DataFrame(
        {
            "Observed ρd (kg/m³)": srmod_df["Density_kg/m3_PT"],
            "Predicted ρd (kg/m³)": phase_ρd(
                θ=srmod_df["θ_R2"], Sr=sr_pred, ρw=1000, Gs=srmod_df["Gs"]
            ),
            "Soil type": srmod_df["Metatype"],
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
def _(np, pl, predND_stats, predρd_stats):
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
    pred_stats_summary.write_csv("data/pred_stats_summary.csv")
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


@app.cell
def _(
    GaussianProcessRegressor,
    Matern,
    StandardScaler,
    WhiteKernel,
    data_02,
    field_results,
    np,
    phase_ρd,
    pl,
    probegp_featureslist,
    root_mean_squared_error,
    srlr_to_sr,
    tqdm,
):
    def bootstrap_rmse(df, field_data, n_bootstrap=1000):
        """Bootstrap analysis with GP - numerically stable version"""
        import warnings

        warnings.filterwarnings("ignore")  # Supprimer les warnings de convergence

        np.random.seed(574764)
        unique_soil_ids = df["Soil_ID"].unique().to_list()
        n_soils = len(unique_soil_ids)
        rmse_sm_list = []
        rmse_nd_list = []

        for i in tqdm(range(n_bootstrap), desc="Bootstrap with GP"):
            resampled_soil_ids = np.random.choice(
                unique_soil_ids, size=n_soils, replace=True
            )

            proctor_subset = data_02.filter(
                pl.col("Soil_ID").is_in(resampled_soil_ids)
            )
            field_subset = df.filter(pl.col("Soil_ID").is_in(resampled_soil_ids))

            if len(proctor_subset) < 10 or len(field_subset) < 5:
                continue

            # Train probe model (GP)
            probegp_features_boot = (
                proctor_subset.select(probegp_featureslist).drop_nulls().to_numpy()
            )
            probegp_targets_boot = (
                proctor_subset.select(["WLR"]).drop_nulls().to_numpy()
            )

            probegp_scaler_boot = StandardScaler()
            probegp_features_boot_sc = probegp_scaler_boot.fit_transform(
                probegp_features_boot
            )

            probegp_model_boot = GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 20.0)
                )
                + WhiteKernel(noise_level=0.1),
                alpha=0.03141,
                n_restarts_optimizer=2,  # Réduit pour vitesse
                normalize_y=True,
                copy_X_train=False,
                random_state=985793 + i,
            )
            probegp_model_boot.fit(probegp_features_boot_sc, probegp_targets_boot)

            # Predict WLR_R1
            field_with_probe = field_results.filter(
                pl.col("Soil_ID").is_in(resampled_soil_ids)
            ).rename({"Probe_before_flood": "Probe"})
            probe_field_features = (
                field_with_probe.select(probegp_featureslist)
                .drop_nulls()
                .to_numpy()
            )
            probe_field_features_sc = probegp_scaler_boot.transform(
                probe_field_features
            )
            WLR_R1_boot = probegp_model_boot.predict(probe_field_features_sc)

            # Train Sr model (GP)
            field_other_features = (
                field_subset.select(["d85", "cu", "Gs"]).drop_nulls().to_numpy()
            )
            field_features_boot = np.column_stack(
                [WLR_R1_boot, field_other_features]
            )
            field_targets_boot = (
                field_subset.select(["SrLR_R2"]).drop_nulls().to_numpy()
            )

            srlr_scaler_boot = StandardScaler()
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
                random_state=106638 + i,
            )
            srlr_model_boot.fit(field_features_boot_sc, field_targets_boot)

            # Predict
            pred_srlr_boot = srlr_model_boot.predict(field_features_boot_sc)
            pred_sr_boot = srlr_to_sr(pred_srlr_boot)

            θ_R2_boot = (
                field_subset.select(["θ_R2"]).drop_nulls().to_numpy().flatten()
            )
            Gs_boot = field_subset.select(["Gs"]).drop_nulls().to_numpy().flatten()
            ρd_observed_boot = (
                field_subset.select(["Density_kg/m3_PT"])
                .drop_nulls()
                .to_numpy()
                .flatten()
            )

            ρd_sm_boot = phase_ρd(
                θ=θ_R2_boot, Sr=pred_sr_boot.flatten(), ρw=1000, Gs=Gs_boot
            )
            rmse_sm = root_mean_squared_error(ρd_observed_boot, ρd_sm_boot)
            rmse_sm_list.append(rmse_sm)

            # ND bootstrap
            field_nd_boot = field_data.filter(
                pl.col("Soil_ID").is_in(resampled_soil_ids.tolist())
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


    def loocv_rmse(proctor_df, field_df):
        """Leave-one-soil-out CV with GP"""
        import warnings

        warnings.filterwarnings("ignore")

        unique_soil_ids = field_df["Soil_ID"].unique().to_list()
        all_squared_errors = []

        for i, test_soil_id in tqdm(
            enumerate(unique_soil_ids, desc="LOOCV with GP")
        ):
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

            probegp_scaler_cv = StandardScaler()
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

            srlr_scaler_cv = StandardScaler()
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
            pred_sr_test = srlr_to_sr(pred_srlr_test)

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
                θ=θ_R2_test, Sr=pred_sr_test.flatten(), ρw=1000, Gs=Gs_test
            )
            squared_errors = (ρd_observed_test - ρd_sm_test) ** 2
            all_squared_errors.extend(squared_errors)

        return (
            np.sqrt(np.mean(all_squared_errors)) if all_squared_errors else np.nan
        )
    return bootstrap_rmse, loocv_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run bootstrap analysis""")
    return


@app.cell
def _(
    bootstrap_rmse,
    data_02,
    field,
    loocv_rmse,
    np,
    phase_ρd,
    root_mean_squared_error,
    srlr_to_sr,
    srmod_df,
):
    rmse_sm_bootstrap, rmse_nd_bootstrap = bootstrap_rmse(
        srmod_df, field, n_bootstrap=1000
    )
    ci_lower_sm = np.percentile(rmse_sm_bootstrap, 2.5)
    ci_upper_sm = np.percentile(rmse_sm_bootstrap, 97.5)
    ci_lower_nd = np.percentile(rmse_nd_bootstrap, 2.5)
    ci_upper_nd = np.percentile(rmse_nd_bootstrap, 97.5)

    rmse_loocv = loocv_rmse(data_02, srmod_df)

    ρd_sm_current = phase_ρd(
        θ=srmod_df["θ_R2"],
        Sr=srlr_to_sr(srmod_df["SrLR_R2"]),
        ρw=1000,
        Gs=srmod_df["Gs"],
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
    print(f"LOOCV SM RMSE: {rmse_loocv:.1f}")
    print(f"Current SM RMSE: {rmse_current_sm:.1f}")
    print(f"Current ND RMSE: {rmse_current_nd:.1f}")
    return (
        ci_lower_nd,
        ci_lower_sm,
        ci_upper_nd,
        ci_upper_sm,
        rmse_current_nd,
        rmse_loocv,
    )


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
    rmse_loocv,
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
            "RMSE": [rmse_all_cv, rmse_loocv, rmse_current_nd],
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
def _(accuracy_summary):
    # Save the summary table to CSV
    accuracy_summary.write_csv("data/accuracy_summary.csv")
    print("Accuracy summary table saved to 'data/accuracy_summary.csv'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## Example""")
    return


@app.cell
def _(
    WLR_to_VWC,
    np,
    probegp_featuresScaler,
    probegp_model,
    srlr_model,
    srlr_to_sr,
    srmod_featuresScaler,
):
    np.random.seed(579025)  # random.org

    n_samples = 1000

    probe_examplefeatures_R1 = np.array([[10.0, 0.08, 2.73, 2131]])
    probe_examplefeatures_R2 = np.array([[10.0, 0.08, 2.73, 2342]])

    probe_WLR1 = probegp_model.sample_y(
        probegp_featuresScaler.transform(probe_examplefeatures_R1),
        n_samples=n_samples,
    )[0]
    probe_WLR2 = probegp_model.sample_y(
        probegp_featuresScaler.transform(probe_examplefeatures_R2),
        n_samples=n_samples,
    )[0]

    probe_θ1 = WLR_to_VWC(probe_WLR1)
    probe_θ2 = WLR_to_VWC(probe_WLR2)
    srlr_model_examplefeatures = np.stack(
        [probe_examplefeatures_R1[0]] * n_samples
    )
    srlr_model_examplefeatures[:, -1] = probe_WLR1
    sr_samples = srlr_to_sr(
        srlr_model.sample_y(
            srmod_featuresScaler.transform(srlr_model_examplefeatures),
            n_samples=n_samples,
        )
    )
    return n_samples, probe_examplefeatures_R1, probe_θ1, probe_θ2, sr_samples


@app.cell
def _(phase_ρd, probe_examplefeatures_R1, probe_θ2, sr_samples):
    srmod_ρd = phase_ρd(
        θ=probe_θ2, Sr=sr_samples, ρw=1000, Gs=probe_examplefeatures_R1[0, 2]
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
