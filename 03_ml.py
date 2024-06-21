import marimo

__generated_with = "0.6.22"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
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
def __():
    # notebook
    import marimo as mo

    # math
    import polars as pl
    import numpy as np
    from scipy.optimize import minimize

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    # model
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error

    np.random.seed(7235)
    return (
        GaussianProcessRegressor,
        GridSearchCV,
        Matern,
        RBF,
        StandardScaler,
        cross_val_predict,
        cross_val_score,
        mean_squared_error,
        minimize,
        mo,
        np,
        pl,
        plt,
        r2_score,
        root_mean_squared_error,
        sns,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Data

        The `proctor` table includes data from the Proctor tests, `soil` includes the optimized particle-size parameters from the notebook `01_multilevel_rosin.ipynb`, and `field` includes the field data. Some tables are joined, since they share the `Soil_ID` column as key.
        """
    )
    return


@app.cell
def __(pl):
    proctor = pl.read_csv("data/r_proctor.csv")
    soils = pl.read_csv("data/soils_rosin.csv")
    field = pl.read_csv("data/r_field.csv", null_values="NA")
    data = proctor.join(soils, on="Soil_ID", how="left", coalesce=True)
    data = data.with_columns((pl.col("VolWC_%") / 100).alias("VolWC"))
    return data, field, proctor, soils


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Overview of data quantity.")
    return


@app.cell
def __(field, proctor, soils):
    print(
        "Number of proctor samples: " + str(proctor["Proctor_ID"].unique().count())
    )
    print("Number of soil samples: " + str(soils["Soil_ID"].unique().count()))
    print("Number of field samples: " + str(field["Field_ID"].unique().count()))
    return


@app.cell(hide_code=True)
def __(mo):
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
def __(data, np, pl):
    def VWC_to_WLR(x):
        # transform volumetric water content to water log ratio
        return np.log(x / (1 - x))


    def WLR_to_VWC(x):
        # transform water log ratio to volumetric water content
        return np.exp(x) / (1 + np.exp(x))


    data_02 = data.with_columns((VWC_to_WLR(pl.col("VolWC"))).alias("WLR"))
    return VWC_to_WLR, WLR_to_VWC, data_02


@app.cell
def __(mo):
    mo.md(rf"Let's see how the probe reacts to water content log ratios.")
    return


@app.cell
def __(data_02, sns):
    sns.scatterplot(data=data_02, x="Probe", y="WLR", hue="Name")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        Points show a growth curve with diminishing return. It could be modelled using an exponential association, typically using three parameters. In this perspective, each test would be fitted to obtain parameters. The parameters could then be subjected to statistical testing and predictive modelling as a function of $g_1$, $g_2$ and $Gs$. Another option is to fit each individual water content (or $wlr$) to $D_{85}$, $Cu$, $Gs$ and the probe value. The result would be less statistically relevant, but very flexible. The last option would also better identify undocumented regions in the data space, and generate more uncertainty in these regions.

        ### Model probe to WLR directly with a Gaussian process

        We use soil grain parameters **and** the probe measurement to predict WLR.
        """
    )
    return


@app.cell
def __(StandardScaler, data_02):
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
        probegp_df,
        probegp_features,
        probegp_featuresScaler,
        probegp_featureslist,
        probegp_featuressc,
        probegp_targets,
        probegp_targetslist,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"A grid serach helps finding optimal hyperparameters.")
    return


@app.cell
def __(
    GaussianProcessRegressor,
    GridSearchCV,
    Matern,
    probegp_featuressc,
    probegp_targets,
):
    probegp_model_cv = GaussianProcessRegressor(n_restarts_optimizer=0)

    param_grid = {
        "alpha": [0.005, 0.01, 0.05, 0.1],
        "kernel": [Matern(length_scale=1, nu=0.5), Matern(length_scale=1, nu=1.5)],
    }

    grid_search = GridSearchCV(probegp_model_cv, param_grid, cv=5)
    grid_search.fit(probegp_featuressc, probegp_targets)
    grid_search.best_params_
    return grid_search, param_grid, probegp_model_cv


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"The best parameters are not the outcomes of the grid search, but the following.")
    return


@app.cell
def __(
    GaussianProcessRegressor,
    Matern,
    cross_val_score,
    np,
    probegp_featuressc,
    probegp_targets,
):
    # model
    probegp_model = GaussianProcessRegressor(
        kernel=Matern(length_scale=1, nu=1.5),
        n_restarts_optimizer=10,
        alpha=0.05,
        normalize_y=True,
    )

    # cross validation
    r2_scores = cross_val_score(
        probegp_model, probegp_featuressc, probegp_targets, cv=10, scoring="r2"
    )
    print(f"Min -R2: {np.min(r2_scores)}")
    print(f"Median R2: {np.median(r2_scores)}")
    print(f"Max R2: {np.max(r2_scores)}")
    return probegp_model, r2_scores


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Fit the model.")
    return


@app.cell
def __(probegp_featuressc, probegp_model, probegp_targets):
    probegp_model.fit(probegp_featuressc, probegp_targets)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Plot model 1")
    return


@app.cell
def __(
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
            "Observed θ": probegp_model_targets_obs,
            "Predicted θ": probegp_model_targets_pred,
            "Soil type": data["Metatype"],
        }
    )
    abline = [0, 0.35]
    g = sns.jointplot(
        data=plot_data,
        x="Observed θ",
        y="Predicted θ",
        hue="Soil type",
        palette="gray",
    )
    g.ax_joint.plot(abline, abline, "black")
    g.savefig("images/probemodel-obs-pred.png")
    g
    return (
        abline,
        g,
        plot_data,
        probegp_model_targets_obs,
        probegp_model_targets_pred,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Plot all calibration curves.")
    return


@app.cell
def __(
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
    n_probe = 200
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
    return (
        cu_list,
        d85_list,
        data_02_idfeatures_i,
        data_02_idtargets_i,
        gs_list,
        k,
        model1_preds,
        n_probe,
        probe,
        probe_list,
        probegp_model_predfeatures,
        probegp_model_predfeatures_sc,
        probegp_model_predtargets,
        soil_id_k,
        soil_ids,
        soil_ids_list,
        vwc_list,
    )


@app.cell
def __(data_02, model1_preds, np, pl, plt, probegp_featureslist, soil_ids):
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
    fig
    return (
        axs,
        col,
        data_02_features,
        data_02_idfeatures,
        data_02_idtargets,
        data_02_targets,
        fig,
        m,
        n,
        plot_ncol,
        plot_nrow,
        probe_m,
        row,
        soil_id,
        vwc,
    )


@app.cell(hide_code=True)
def __(mo):
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
def __():
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
def __(mo):
    mo.md(rf"Data from table `field`, containing results from field experiments, will be used. We'll need $d_{85}$ and $Cu$, which are joined to the `field` table.")
    return


@app.cell
def __(field, soils):
    field_psd = field.join(
        soils.select(["Soil_ID", "cu", "d85"]),
        on="Soil_ID",
        how="left",
        coalesce=False,
    )
    return field_psd,


@app.cell(hide_code=True)
def __(mo):
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
def __(np):
    def sr_to_srlr(sr):
        return np.log(sr / (1 - sr))


    def srlr_to_sr(srlr, maxsr=0.99):
        return np.minimum(maxsr, np.exp(srlr) / (1 + np.exp(srlr)))
    return sr_to_srlr, srlr_to_sr


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"We create a new data frame, containing the columns we need, then removing all rows containing at least one null (unmeasured) value.")
    return


@app.cell
def __(field_psd):
    field_full = field_psd.select(
        [
            "d85",
            "cu",
            "Gs",
            "Probe_before_flood",
            "Probe_after_1minflood",
            "Density_kg/m3_PT",
            "Metatype",
        ]
    ).drop_nulls()
    return field_full,


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Predict θ_R1 and θ_R2 with one of the models created before.")
    return


@app.cell
def __(field_full):
    probefield_R1 = field_full.select("Probe_before_flood").to_numpy()
    probefield_R2 = field_full.select("Probe_after_1minflood").to_numpy()
    return probefield_R1, probefield_R2


@app.cell
def __(field_full, probegp_featuresScaler, probegp_model):
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
    return (
        WLR_R1,
        WLR_R2,
        probegp_model_fieldfeatures_after,
        probegp_model_fieldfeatures_aftersc,
        probegp_model_fieldfeatures_before,
        probegp_model_fieldfeatures_beforesc,
        probegp_model_fieldfeatureslist,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Gather the info for modelling.")
    return


@app.cell
def __(
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
                pl.Series([1000.0]).alias("ρw"),
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
        ).with_columns(sr_to_srlr(pl.col("Sr_R2")).alias("SrLR_R2"))
    )
    return field_results, θ_R1, θ_R2


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Transforming data.")
    return


@app.cell
def __(StandardScaler, field_results, pl):
    # clean data
    srmod_features = ["WLR_R1", "d85", "cu", "Gs"]
    srmod_target = ["SrLR_R2"]
    srmod_df = (
        field_results.select(
            srmod_features
            + srmod_target
            + ["Density_kg/m3_PT", "θ_R2", "Metatype"]
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
    srmod_featuresrsc = (srmod_featurestr - srmod_featuresmean) / srmod_featuresstd

    # targets
    srmod_targettr = srmod_df[srmod_target].to_numpy()
    return (
        srmod_df,
        srmod_features,
        srmod_featuresScaler,
        srmod_featuresmean,
        srmod_featuresrsc,
        srmod_featuresstd,
        srmod_featurestr,
        srmod_target,
        srmod_targettr,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Looking for the most appropriate `alpha` with grid search cross validation.")
    return


@app.cell
def __(
    GaussianProcessRegressor,
    GridSearchCV,
    Matern,
    srmod_featuresrsc,
    srmod_targettr,
):
    srlr_model_cv = GaussianProcessRegressor(n_restarts_optimizer=0)

    param_grid_sr = {
        "alpha": [0.2, 0.21, 0.22, 0.23, 0.24],
        "kernel": [Matern(length_scale=1, nu=0.5)],
    }

    grid_search_sr = GridSearchCV(srlr_model_cv, param_grid_sr, cv=10)
    grid_search_sr.fit(srmod_featuresrsc, srmod_targettr)
    grid_search_sr.best_params_
    return grid_search_sr, param_grid_sr, srlr_model_cv


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"We then model Sr with a Gaussian process.")
    return


@app.cell
def __(
    GaussianProcessRegressor,
    Matern,
    cross_val_score,
    np,
    srmod_featuresrsc,
    srmod_targettr,
):
    # model
    srlr_model = GaussianProcessRegressor(
        kernel=Matern(length_scale=1, nu=0.5),
        n_restarts_optimizer=10,
        alpha=0.22,
        normalize_y=True,
    )

    # cross validation
    r2_scores_sr = cross_val_score(
        srlr_model, srmod_featuresrsc, srmod_targettr, cv=10, scoring="r2"
    )
    print(f"Min R2 : {np.min(r2_scores_sr)}")
    print(f"Median R2 : {np.median(r2_scores_sr)}")
    print(f"Max R2 : {np.max(r2_scores_sr)}")
    return r2_scores_sr, srlr_model


@app.cell
def __(
    root_mean_squared_error,
    srlr_model,
    srlr_to_sr,
    srmod_featuresrsc,
    srmod_targettr,
):
    srlr_model.fit(srmod_featuresrsc, srmod_targettr)
    srlr_pred = srlr_model.predict(srmod_featuresrsc)
    print(f"RMSE for Sr: {root_mean_squared_error(srmod_targettr, srlr_pred)}")
    sr_pred = srlr_to_sr(srlr_pred)
    return sr_pred, srlr_pred


@app.cell
def __(pl, sns, sr_pred, srlr_to_sr, srmod_df):
    plot_data_sr = pl.DataFrame(
        {
            "Observed Sr": srlr_to_sr(srmod_df["SrLR_R2"].to_numpy()),
            "Predicted Sr": sr_pred,
            "Soil type": srmod_df["Metatype"],
        }
    )
    abline_sr = [0.3, 1.0]
    gsr = sns.jointplot(
        data=plot_data_sr,
        x="Observed Sr",
        y="Predicted Sr",
        hue="Soil type",
        palette="gray",
    )
    gsr.ax_joint.plot(abline_sr, abline_sr, "black")
    gsr.savefig("images/srmodel-obs-pred.png")
    gsr
    return abline_sr, gsr, plot_data_sr


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Once Sr is modelled, predictions can be expressed in terms of $ρ_d$.")
    return


@app.cell
def __(phase_ρd, pl, sns, sr_pred, srmod_df):
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
    return abline_ρd, gρd, plot_data_ρd


@app.cell
def __(pl, plot_data_ρd):
    predρd_stats = plot_data_ρd.with_columns(
        (
            (pl.col("Observed ρd (kg/m³)") - plot_data_ρd["Predicted ρd (kg/m³)"])
            ** 2
        ).alias("squared_errors")
    ).with_columns(pl.Series("Device", ["Sherbrooke Method"] * len(plot_data_ρd)))
    return predρd_stats,


@app.cell(hide_code=True)
def __(mo):
    mo.md("Compared to nucleodensimeter...")
    return


@app.cell
def __(abline_ρd, field, pl, sns):
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
    return abline_ND, gND, plot_data_ND


@app.cell
def __(pl, plot_data_ND):
    predND_stats = plot_data_ND.with_columns(
        (
            (pl.col("Observed ρd (kg/m³)") - plot_data_ND["Predicted ρd (kg/m³)"])
            ** 2
        ).alias("squared_errors")
    ).with_columns(pl.Series("Device", ["Nucleodensimeter"] * len(plot_data_ND)))
    return predND_stats,


@app.cell(hide_code=True)
def __(mo):
    mo.md("Summary statistics for accuracy assessment.")
    return


@app.cell
def __(np, pl, predND_stats, predρd_stats):
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
    return (
        pred_stats,
        pred_stats_all,
        pred_stats_per_soil,
        pred_stats_summary,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"## Example")
    return


@app.cell
def __(
    WLR_to_VWC,
    np,
    probegp_featuresScaler,
    probegp_model,
    srlr_model,
    srlr_to_sr,
    srmod_featuresScaler,
):
    n_samples = 1000

    probe_examplefeatures_R1 = np.array([[1.0, 0.05, 2.72, 2100]])
    probe_examplefeatures_R2 = np.array([[1.0, 0.05, 2.72, 2300]])

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
    return (
        n_samples,
        probe_WLR1,
        probe_WLR2,
        probe_examplefeatures_R1,
        probe_examplefeatures_R2,
        probe_θ1,
        probe_θ2,
        sr_samples,
        srlr_model_examplefeatures,
    )


@app.cell
def __(phase_ρd, probe_examplefeatures_R1, probe_θ2, sr_samples):
    srmod_ρd = phase_ρd(
        θ=probe_θ2, Sr=sr_samples, ρw=1000, Gs=probe_examplefeatures_R1[0, 2]
    )
    return srmod_ρd,


@app.cell
def __(n_samples, np, pl, plt, probe_θ1, probe_θ2, sns):
    probe_θ2
    probe_θ_df = pl.DataFrame(
        {
            "Value": np.concatenate([probe_θ1, probe_θ2]),
            "Probe": ["probe θ1"] * n_samples + ["probe θ2"] * n_samples,
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
        palette="grey",
        edgecolor="#333",
        alpha=0.7,
    )
    plt.savefig("images/vwc_ditr.png")
    vwc_ditr
    return probe_θ_df, vwc_ditr


@app.cell
def __(np, plt, srmod_ρd):
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
    rho_distr
    return density_limit, prob_sup, rho_distr, srmod_ρd_flat


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
