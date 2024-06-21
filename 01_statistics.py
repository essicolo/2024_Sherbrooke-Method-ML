import marimo

__generated_with = "0.6.19"
app = marimo.App()


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import pickle

    mo.md(
        """
        # Statistical modelling
        """
    )
    return mo, pickle


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"## Load librairies")
    return


@app.cell
def __():
    # data and math
    import pandas as pd
    import polars as pl
    import numpy as np

    # plot
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-colorblind")
    import altair as alt
    alt.data_transformers.disable_max_rows()

    # statistics
    from scipy import stats
    from scipy.optimize import curve_fit
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pymc as pm
    import pytensor.tensor as pt  # for pymc
    return alt, curve_fit, np, pd, pl, plt, pm, pt, sm, smf, stats


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"## Load and inspect data")
    return


@app.cell
def __(pl):
    grainsize = pl.read_csv("data/r_grainsize.csv")
    soils_a = pl.read_csv("data/r_soils.csv")
    proctor_a = pl.read_csv("data/r_proctor.csv")
    proctor_a = proctor_a.rename({"VolWC_%": "theta"})
    proctor_a = proctor_a.with_columns(pl.col("theta") / 100)
    return grainsize, proctor_a, soils_a


@app.cell
def __(alt, proctor_a):
    alt.Chart(proctor_a).mark_point(color="black", fill="black").encode(
        x=alt.X("Probe", scale=alt.Scale(zero=False), title="Probe value"),
        y=alt.Y(
            "theta",
            scale=alt.Scale(zero=False),
            title="Volumetric water content, θ",
        ),
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"## PSD")
    return


@app.cell
def __(alt, grainsize):
    alt.Chart(grainsize).mark_line().encode(
        x=alt.X("Mesh:Q", scale=alt.Scale(type="log", domain=(0.01, 1000))),
        y="Perc_passing:Q",
        color="Soil_ID:N",
    ).properties(width=600, height=300)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Models found in the literature, reported by Esmaeelnejad et al. (2016), fig 1.
        https://link.springer.com/article/10.1007/s40808-016-0220-9

        <img src='images/40808_2016_220_Fig1_HTML.webp'>
        """
    )
    return


@app.cell
def __(np):
    def weibull(d, a, b, c):
        return a - np.exp(-((d / b) ** c))


    def vgpsd(d, g1, g2, k3):
        return (1 + (g1 / d) ** g2) ** (-k3)


    def rosin(d, g1, g2):
        return 1 - np.exp(-g1 * d**g2)


    def fredlund(d, a, m, n, dr):
        return (1 / (np.log(np.exp(1) + (a / d) ** n) ** m)) * (
            1 - np.log(1 + dr / d) / np.log(dr / 0.001)
        )
    return fredlund, rosin, vgpsd, weibull


@app.cell
def __(curve_fit, grainsize, np, pl, rosin):
    soil_ids = grainsize["Soil_ID"].unique().to_list()
    weibull_param = {"Soil_ID": [], "a": [], "b": [], "c": [], "rmse": []}
    rosin_param = {"Soil_ID": [], "g1": [], "g2": [], "rmse": []}
    vgpsd_param = {"Soil_ID": [], "g1": [], "g2": [], "k3": [], "rmse": []}
    fredlund_param = {
        "Soil_ID": [],
        "a": [],
        "m": [],
        "n": [],
        "dr": [],
        "rmse": [],
    }

    for soil_id in soil_ids:
        xy = (
            grainsize.filter(pl.col("Soil_ID") == soil_id)
            .select(["Mesh", "Perc_passing"])
            .with_columns(pl.col("Perc_passing") / 100)
            .to_numpy()
        )
        if ~np.any(np.isnan(xy)):
            # weibull_cf = curve_fit(weibull, xy[:, 0], xy[:, 1])
            # weibull_param['Soil_ID'].append(soil_id)
            # weibull_param['a'].append(weibull_cf[0][0])
            # weibull_param['b'].append(weibull_cf[0][1])
            # weibull_param['c'].append(weibull_cf[0][2])
            # weibull_pred = weibull(d=xy[:, 0], a=weibull_cf[0][0], b=weibull_cf[0][1], c=weibull_cf[0][2])
            # weibull_param['rmse'].append(np.mean(np.sqrt((xy[:, 1] - weibull_pred)**2)))

            rosin_cf = curve_fit(
                rosin, xy[:, 0], xy[:, 1]
            )  # , bounds=([5, 5], [80, 80]), method='dogbox'
            rosin_param["Soil_ID"].append(soil_id)
            rosin_param["g1"].append(rosin_cf[0][0])
            rosin_param["g2"].append(rosin_cf[0][1])
            rosin_pred = rosin(d=xy[:, 0], g1=rosin_cf[0][0], g2=rosin_cf[0][1])
            rosin_param["rmse"].append(
                np.mean(np.sqrt((xy[:, 1] - rosin_pred) ** 2))
            )

            # vgpsd_cf = curve_fit(vgpsd, xy[:, 0], xy[:, 1])
            # vgpsd_param['Soil_ID'].append(soil_id)
            # vgpsd_param['g1'].append(vgpsd_cf[0][0])
            # vgpsd_param['g2'].append(vgpsd_cf[0][1])
            # vgpsd_param['k3'].append(vgpsd_cf[0][2])
            # vgpsd_pred = vgpsd(d=xy[:, 0], g1=vgpsd_cf[0][0], g2=vgpsd_cf[0][1], k3=vgpsd_cf[0][2])
            # vgpsd_param['rmse'].append(np.mean(np.sqrt((xy[:, 1] - vgpsd_pred)**2)))

            # fredlund_cf = curve_fit(
            #     fredlund, xy[:, 0], xy[:, 1],
            #     p0=[0.1, 2, 2, 1000], method='lm'
            # )
            # fredlund_param['Soil_ID'].append(soil_id)
            # fredlund_param['a'].append(fredlund_cf[0][0])
            # fredlund_param['m'].append(fredlund_cf[0][1])
            # fredlund_param['n'].append(fredlund_cf[0][2])
            # fredlund_param['dr'].append(fredlund_cf[0][3])
    return (
        fredlund_param,
        rosin_cf,
        rosin_param,
        rosin_pred,
        soil_id,
        soil_ids,
        vgpsd_param,
        weibull_param,
        xy,
    )


@app.cell
def __(pl, rosin_param):
    rosin_param_df_a = pl.DataFrame(rosin_param)
    print("rosin", rosin_param_df_a["rmse"].mean())
    return rosin_param_df_a,


@app.cell
def __(alt, grainsize, np, pl, rosin, rosin_param_df_a):
    diameter_a = np.logspace(-3, 3, 100)
    psd_est = (
        rosin(diameter_a, rosin_param_df_a[0, "g1"], rosin_param_df_a[0, "g2"])
        * 100
    )

    line_grainsize = (
        alt.Chart(grainsize.filter(pl.col("Soil_ID") == 1))
        .mark_point()
        .encode(
            x=alt.X("Mesh:Q", scale=alt.Scale(type="log", domain=(0.001, 1000))),
            y="Perc_passing:Q",
            color=alt.value("black"),
            fill=alt.value("black"),
        )
    )

    # Create the line chart for the estimated data
    line_estimated = (
        alt.Chart(pl.DataFrame({"Mesh": diameter_a, "Perc_passing": psd_est}))
        .mark_line()
        .encode(x="Mesh:Q", y="Perc_passing:Q", color=alt.value("red"))
    )

    # Combine the charts
    chart = (line_estimated + line_grainsize).properties(width=600, height=300)

    chart
    return chart, diameter_a, line_estimated, line_grainsize, psd_est


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Join Rosin's parameters to the `proctor` table.

        ### Transform Rosin's grain-size parameters to common grain-size coefficients used in geotechnics

        $$
        p = 1 - exp(-g_1 d^{g_2})
        $$

        $$
        exp(-g_1 d^{g_2}) = 1 - p
        $$

        $$
        -g_1 d^{g_2} = log(1 - p)
        $$

        $$
        d = \left(- \frac{log(1 - p)}{g_1} \right) ^{1/g_2}
        $$

        $$
        d_{85} = \left(- \frac{log(0.15)}{g_1} \right) ^{1/g_2}
        $$


        $$
        C_u = \frac{\left(- \frac{log(1 - 0.6)}{g_1} \right) ^{1/g_2}}{\left(- \frac{log(1 - 0.1)}{g_1} \right) ^{1/g_2}}
        $$

        $$
        C_u = \left( \frac{- \frac{log(1 - 0.6)}{g_1}}{- \frac{log(1 - 0.1)}{g_1}} \right)^{1/g_2}
        $$

        $$
        C_u = \left( \frac{ log(1 - 0.6)}{log(1 - 0.1)} \right)^{1/g_2}
        $$

        $$
        C_u = \left( \frac{log(0.4)}{log(0.9)}\right)^{1/g_2}
        $$

        $$
        g_2 =  \frac {log \left(  \frac{log(0.9)}{log(0.4)}\right)}{log(C_u)}
        $$

        $$
        g_1 = \frac{log(0.15)}{d_{85}^{g_2}}
        $$
        """
    )
    return


@app.cell
def __(np, pl, rosin_param_df_a):
    def rosin_to_geotech(g1, g2):
        d85 = (-np.log(0.15) / g1) ** (1 / g2)
        cu = (np.log(0.9) / np.log(0.4)) ** (1 / g2)
        return d85, cu


    def geotech_to_rosin(d85, cu):
        g2 = np.log(np.log(0.9) / np.log(0.4)) / np.log(cu)
        g1 = -np.log(0.15) / (d85**g2)
        return g1, g2


    g1_values = rosin_param_df_a["g1"].to_numpy()
    g2_values = rosin_param_df_a["g2"].to_numpy()
    d85_values, cu_values = rosin_to_geotech(g1_values, g2_values)

    rosin_param_df_b = rosin_param_df_a.with_columns(
        pl.Series(d85_values).alias("d85"), pl.Series(cu_values).alias("cu")
    )
    return (
        cu_values,
        d85_values,
        g1_values,
        g2_values,
        geotech_to_rosin,
        rosin_param_df_b,
        rosin_to_geotech,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Join Rosin's parameters and fitted d85 and cu.")
    return


app._unparsable_cell(
    r"""
    proctor_b = proctor_a.join(rosin_param_df_b, how=\"left\", on=\"Soil_ID\"coalesce=True)
    soils_b = soils_a.join(rosin_param_df_b, how=\"left\", on=\"Soil_ID\", coalesce=True)
    soils_b.write_csv(\"data/soils_rosin.csv\")
    """,
    name="__"
)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Évaluer les plages de paramètres.")
    return


@app.cell
def __(pl, proctor_b):
    kvar_ranges = (
        proctor_b.melt(id_vars="Soil_ID", value_vars=["d85", "cu"])
        .group_by("variable")
        .agg([pl.min("value").alias("min"), pl.max("value").alias("max")])
    )
    kvar_ranges
    return kvar_ranges,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Effect of parameters on the particle size distribution.")
    return


@app.cell
def __(np, pl, rosin_param_df_b):
    nk_ = 50

    g1_df = pl.DataFrame(
        dict(
            var_on="g1",
            g1=np.logspace(-3, 2, nk_),
            g2=rosin_param_df_b["g2"].mean(),
        )
    )

    g2_df = pl.DataFrame(
        dict(
            var_on="g2",
            g1=rosin_param_df_b["g1"].mean(),
            g2=np.linspace(0.25, 3, nk_),
        )
    )

    psd_constants = g1_df.vstack(g2_df)
    return g1_df, g2_df, nk_, psd_constants


@app.cell
def __(alt, np, pl, plt, psd_constants, rosin):
    nd_ = 300
    d = np.logspace(-3, 3, nd_)

    kvars = psd_constants.select("var_on").unique().to_numpy().flatten().tolist()

    p_ = []

    k_description = [
        "lié au diamètre à l'inflexion",
        "lié à l'uniformité de la distribution",
    ]

    plt.tight_layout()
    for _i, _k in enumerate(kvars):
        psd_k = psd_constants.filter(pl.col("var_on") == _k)
        psd_kX = pl.DataFrame(dict(k=[0.0], d=[0.0], perc_passing=[0.0]))
        # print(psd_k)
        for _j in range(psd_k.shape[0]):
            psd_kX = psd_kX.vstack(
                pl.DataFrame(
                    dict(
                        k=psd_k[_j, _k],
                        d=d,
                        perc_passing=rosin(
                            d=d, g1=psd_k[_j, "g1"], g2=psd_k[_j, "g2"]
                        )
                        * 100,
                    )
                )
            )
        p_.append(
            alt.Chart(psd_kX[1:, :])
            .mark_line()
            .encode(
                x=alt.X(
                    "d",
                    axis=alt.Axis(title="Diamètre (mm)"),
                    scale=alt.Scale(type="log", domain=[0.001, 1000]),
                ),
                y=alt.Y(
                    "perc_passing", axis=alt.Axis(title="Pourcentage passant")
                ),
                color=alt.Color("k", title=""),
            )
            .properties(title=_k + " : " + k_description[_i])
        )

    (p_[0] | p_[1]).resolve_scale(color="independent")
    return d, k_description, kvars, nd_, p_, psd_k, psd_kX


@app.cell
def __(alt, rosin_param_df_b, soils_b):
    kvar_soils = (
        soils_b.join(rosin_param_df_b, how="left", on="Soil_ID")
        .select(["Name", "Soil_ID", "d85", "cu"])
        .melt(id_vars=["Name", "Soil_ID"], value_vars=["d85", "cu"])
    )


    (
        alt.Chart(kvar_soils)
        .encode(y="Name:N")
        .mark_text(align="center", baseline="middle")
        .encode(x=alt.X("value:Q", scale=alt.Scale(type="log")), text="Soil_ID:N")
        .properties(width=400, height=400)
        .facet("variable:N", columns=2)
        .resolve_scale(x="independent", y="independent")
        .configure_axisY(grid=True)
    )
    return kvar_soils,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The grain size curves as well as the zones were represented in R, given the unavailability of a function similar to `gghighlight` in Python.

        ![](images/granulos.png)

        Sol 29, very coarse, is out of category. Soils 3 and 18, whose grain sizes are finer, have parameters that do not stand out from the rest.

        ## Mixed quadratic model, frequency mode
        """
    )
    return


@app.cell
def __(pl, proctor_b, smf):
    proctor_MG112 = proctor_b.filter(pl.col("Metatype") == "MG112")

    freq_qmm = smf.mixedlm(
        "theta ~ Probe + I(Probe**2) + d85 + cu + Gs",  #
        proctor_MG112.to_pandas(),
        groups=proctor_MG112["Site"],
    ).fit()
    freq_qmm.summary()
    return freq_qmm, proctor_MG112


@app.cell
def __():
    def quadmod(x, a, b, c):
        return a + b * x + c * x**2
    return quadmod,


@app.cell
def __(freq_qmm, np, plt, proctor_MG112, quadmod):
    x_seq_a = np.linspace(1900, 2900, 100)
    y_quad = quadmod(
        x_seq_a,
        freq_qmm.fe_params[0],
        freq_qmm.fe_params[1],
        freq_qmm.fe_params[2],
    )
    plt.plot(x_seq_a, y_quad)
    plt.plot(proctor_MG112["Probe"], proctor_MG112["theta"], ".")
    return x_seq_a, y_quad


@app.cell
def __(freq_qmm):
    freq_qmm.fe_params
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Exponential association model

        Nonlinear modeling is statistically more difficult to perform with traditional methods. I switch to Bayesian mode to obtain a more direct interpretation of the posterior distribution of the parameters. The average model should look like the following.
        """
    )
    return


@app.cell
def __(alt, np, pl, proctor_b):
    def ae(x, xref, ymin, ymax, slope):
        y = (ymax - ymin) * (1 - np.exp(-slope * (x - xref))) + ymin
        return y


    ae_estimate = pl.DataFrame(dict(x=np.linspace(1900, 2800, 50))).with_columns(
        (ae(pl.col("x"), 2000, 0.06, 0.22, 0.003)).alias("y")
    )

    p = (
        alt.Chart(proctor_b.to_pandas())
        .mark_point(color="black", fill="black", size=2)
        .encode(x=alt.X("Probe", scale=alt.Scale(zero=False)), y="theta")
    )
    l = (
        alt.Chart(ae_estimate.to_pandas())
        .mark_line(color="red")
        .encode(x="x", y="y")
    )
    (p + l).properties(width=600)
    return ae, ae_estimate, l, p


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The model is defined according to vague a priori, an approach which makes it possible to limit the exploration of the model domain without excessively constraining it. The model includes fixed effects such as probe measurement, particle size parameters and Gs, as well as random effects linked to the experimental site. Fixed and random effects are considered additive, that is, they cause a vertical shift in the calibration curve. The University of Sherbrooke site serving as a reference, the site effect will be calculated as a difference compared to this site.

        Coefficients of unscaled variables are difficult to initiate. We overcome the problem by scaling d85, cu and gs to zero mean and unit variable. This scaling will also make coefficients comparable between them.
        """
    )
    return


@app.cell
def __(pd, proctor_b):
    x_b = proctor_b["Probe"].to_numpy()
    d85_b = proctor_b["d85"].to_numpy()
    d85sc = (
        (proctor_b["d85"] - proctor_b["d85"].mean()) / proctor_b["d85"].std()
    ).to_numpy()
    cu_b = proctor_b["cu"].to_numpy()
    cusc = (
        (proctor_b["cu"] - proctor_b["cu"].mean()) / proctor_b["cu"].std()
    ).to_numpy()
    gs_b = proctor_b["Gs"].to_numpy()
    gssc = (
        (proctor_b["Gs"] - proctor_b["Gs"].mean()) / proctor_b["Gs"].std()
    ).to_numpy()
    y_ = proctor_b["theta"].to_numpy()
    sites_df = pd.get_dummies(proctor_b["Site"]).drop("Université de Sherbrooke", axis=1)
    sites = (
        sites_df
        .astype(int)
        .values
    )
    xref = 2000
    n_tune = 1000
    n_draws = 5000
    return (
        cu_b,
        cusc,
        d85_b,
        d85sc,
        gs_b,
        gssc,
        n_draws,
        n_tune,
        sites,
        sites_df,
        x_b,
        xref,
        y_,
    )


@app.cell
def __(cusc, d85sc, gssc, n_draws, n_tune, pm, pt, sites, x_b, xref, y_):
    coefficient_std = 10
    with pm.Model() as aemodel:
        outcome_sd = pm.HalfNormal("outcome_sd", sigma=0.2)
        ymin = pm.TruncatedNormal("ymin", mu=0.06, sigma=0.1, lower = 0, upper = 0.2)
        ymax = pm.TruncatedNormal("ymax", mu=0.22, sigma=0.1, lower = ymin, upper = 0.5)
        slope = pm.TruncatedNormal("slope", mu=0.003, sigma = 0.05, lower = 0, upper=None)

        # additive alteration to ymin
        ymin_d85 = pm.Normal("ymin_d85", 0, coefficient_std)
        ymin_cu = pm.Normal("ymin_cu", 0, coefficient_std)
        ymin_gs = pm.Normal("ymin_gs", 0, coefficient_std)
        ymin_ = ymin + ymin_d85*d85sc + ymin_cu*cusc + ymin_gs*gssc
        #ymin_ = ymin + ymin_d85*d85_b + ymin_cu*cu_b + ymin_gs*gs_b

        # additive alteration to ymax
        ymax_d85 = pm.Normal("ymax_d85", 0, coefficient_std)
        ymax_cu = pm.Normal("ymax_cu", 0, coefficient_std)
        ymax_gs = pm.Normal("ymax_gs", 0, coefficient_std)
        ymax_ = ymax + ymax_d85*d85sc + ymax_cu*cusc + ymax_gs*gssc
        #ymax_ = ymax + ymax_d85*d85_b + ymax_cu*cu_b + ymax_gs*gs_b

        # additive alteration to slope
        slope_d85 = pm.Normal("slope_d85", 0, coefficient_std)
        slope_cu = pm.Normal("slope_cu", 0, coefficient_std)
        slope_gs = pm.Normal("slope_gs", 0, coefficient_std)
        slope_ = slope + slope_d85*d85sc + slope_cu*cusc + slope_gs*gssc
        #slope_ = slope + slope_d85*d85_b + slope_cu*cu_b + slope_gs*gs_b

        # random effect
        site_coeffs = pm.Normal("sites", 0, sigma=10, shape=sites.shape[1])

        # model
        model = (ymax_ - ymin_) * (1 - pt.exp(-slope_ * (x_b - xref))) + ymin_ + pt.dot(sites, site_coeffs)
        likelihood = pm.TruncatedNormal("y", mu=model, sigma=outcome_sd, lower=0, observed=y_)
        aetrace = pm.sample(draws=n_draws, tune=n_tune, init='jitter+adapt_diag', return_inferencedata=True)
    return (
        aemodel,
        aetrace,
        coefficient_std,
        likelihood,
        model,
        outcome_sd,
        site_coeffs,
        slope,
        slope_,
        slope_cu,
        slope_d85,
        slope_gs,
        ymax,
        ymax_,
        ymax_cu,
        ymax_d85,
        ymax_gs,
        ymin,
        ymin_,
        ymin_cu,
        ymin_d85,
        ymin_gs,
    )


@app.cell
def __(mo):
    mo.md("The model takes a lot of time to sample, so it is saved with pickle for other uses.")
    return


@app.cell
def __(aetrace, pickle):
    with open("aetrace.pkl", "wb") as buff:
        pickle.dump({"aetrace": aetrace}, buff)
    return buff,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"The posterior distributions are presented in this general overview of the traces of the explored parameters. A good model will show traces that are relatively constant on the right side.")
    return


@app.cell
def __(aetrace, plt, pm):
    pm.plot_trace(aetrace, combined=True)
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"I record the summary of the model in a table where the compatibility intervals (commonly referred to as confidence intervals) are presented at 95%, a typical (but arbitrary) level used for statistical significance.")
    return


@app.cell
def __(aetrace, pm):
    aeresults = pm.summary(aetrace, hdi_prob=0.95, round_to="none")
    return aeresults,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"The random effects of the model encompass site effects and incorporate variability that is not explained by the experimental parameters.")
    return


@app.cell
def __(aeresults, alt, sites_df):
    sites_results = aeresults.loc[aeresults.index.str.startswith("site"), :]
    sites_results = sites_results.rename(
        dict(zip(sites_results.index, sites_df.columns))
    )

    points_results = (
        alt.Chart(sites_results.reset_index())
        .mark_point(color="black", filled=True)
        .encode(
            x=alt.X(
                "mean", axis=alt.Axis(title="Confidence interval at 95% level")
            ),
            y=alt.Y("index", axis=alt.Axis(title="")),
            tooltip="mean",
        )
    )
    bars_results = (
        alt.Chart(sites_results.reset_index())
        .mark_errorbar()
        .encode(
            x="hdi_2.5%",
            x2="hdi_97.5%",
            y="index",
            # tooltip = ['hdi_2.5%', 'hdi_97.5%', 'index']
        )
    )

    (points_results + bars_results).properties(width=600).interactive()
    return bars_results, points_results, sites_results


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Random effects generally have zero mean. But the Bayesian version is more flexible. To ensure an adequate statistical model, the equation should add the average of the random effects to the result.")
    return


@app.cell
def __(sites_results):
    ranef_mean = sites_results["mean"].mean()
    ranef_mean
    return ranef_mean,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"Fixed effects are the effects identifiable by experimentation. The parameters noted here psd[0], psd[1] and psd[2] are respectively the parameters g1, g2 and k3 of the soil grain size. Note that the importance of parameters in the model cannot be judged by the magnitude of a coefficient.")
    return


@app.cell
def __(aeresults):
    aeresults
    return


@app.cell
def __(aeresults):
    fixef = aeresults.loc[
        [
            "ymin",
            "ymax",
            "slope",
            "ymin_d85",
            "ymin_cu",
            "ymin_gs",
            "ymax_d85",
            "ymax_cu",
            "ymax_gs",
            "slope_d85",
            "slope_cu",
            "slope_gs",
        ],
        :,
    ]
    fixef["effect category"] = (
        ["meta effect"] * 3
        + ["ymin effect"] * 3
        + ["ymax effect"] * 3
        + ["slope effect"] * 3
    )
    fixef["effect variable"] = [
        "meta",
        "meta",
        "meta",
        "d85",
        "cu",
        "gs",
        "d85",
        "cu",
        "gs",
        "d85",
        "cu",
        "gs",
    ]
    fixef = fixef.reset_index()
    fixef
    return fixef,


@app.cell
def __(aetrace, pm):
    aetrace_densityplot = pm.plot_density(aetrace, var_names=["ymin_d85", "ymin_cu", "ymin_gs"])
    aetrace_densityplot
    return aetrace_densityplot,


@app.cell
def __(fixef, pl):
    pl.DataFrame(fixef).filter(pl.col("effect variable") != "meta")
    return


@app.cell
def __(alt, fixef, pl):
    base = alt.Chart(
        pl.DataFrame(fixef).filter(pl.col("effect variable") != "meta")
    ).encode(y=alt.Y("index:N", axis=alt.Axis(title="")))
    error_bars = base.mark_errorbar().encode(
        x=alt.X("hdi_2.5%:Q", axis=alt.Axis(title="")),
        x2="hdi_97.5%:Q",
    )
    points = base.mark_point(color="black", filled=True).encode(
        x=alt.X("mean:Q", axis=alt.Axis(title="Confidence interval at 95% level")),
    )
    combined = alt.layer(error_bars, points).properties(width=300, height=120)
    faceted_chart = combined.facet(
        facet="effect category:N", columns=4
    ).resolve_scale(x="independent", y="independent")

    faceted_chart
    return base, combined, error_bars, faceted_chart, points


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
