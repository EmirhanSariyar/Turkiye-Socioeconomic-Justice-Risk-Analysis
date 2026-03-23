import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MAPS_DIR = BASE_DIR / "data" / "external" / "maps"

MASTER_PATH = PROCESSED_DIR / "province_year_master_2011_2021.csv"
MODELING_PATH = PROCESSED_DIR / "province_year_modeling_2011_2021.csv"
RAW_SGK_PATH = RAW_DIR / "sgk_active_insured_2009_2024.csv"
RAW_MEB_PATH = RAW_DIR / "meb_secondary_gross_enrollment_2011_2025.csv"
GEOJSON_PATH = MAPS_DIR / "lvl1-TR.geojson"
BENCHMARK_SUMMARY_PATH = BASE_DIR / "models" / "benchmark_summary.csv"
BENCHMARK_RESULTS_PATH = BASE_DIR / "models" / "benchmark_results.json"
MODEL_PREDICTIONS_PATH = BASE_DIR / "models" / "model_predictions.csv"

TURKISH_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
PROVINCE_NAME_ALIASES = {
    "afyon": "afyonkarahisar",
    "adiyaman": "adiyaman",
    "adyaman": "adiyaman",
    "agri": "agri",
    "agr": "agri",
    "aydin": "aydin",
    "aydn": "aydin",
    "balikesir": "balikesir",
    "balkesir": "balikesir",
    "bartin": "bartin",
    "bartn": "bartin",
    "canakkale": "canakkale",
    "cankiri": "cankiri",
    "cankr": "cankiri",
    "corum": "corum",
    "diyarbakir": "diyarbakir",
    "diyarbakr": "diyarbakir",
    "duzce": "duzce",
    "elazig": "elazig",
    "elazg": "elazig",
    "eskisehir": "eskisehir",
    "gumushane": "gumushane",
    "igdir": "igdir",
    "igdr": "igdir",
    "izmir": "izmir",
    "kahramanmaras": "kahramanmaras",
    "kmaras": "kahramanmaras",
    "karabuk": "karabuk",
    "kirikkale": "kirikkale",
    "kinkkale": "kirikkale",
    "krkkale": "kirikkale",
    "kirklareli": "kirklareli",
    "krklareli": "kirklareli",
    "kirsehir": "kirsehir",
    "krsehir": "kirsehir",
    "kutahya": "kutahya",
    "mugla": "mugla",
    "mus": "mus",
    "nevsehir": "nevsehir",
    "nigde": "nigde",
    "sanliurfa": "sanliurfa",
    "sanlurfa": "sanliurfa",
    "sirnak": "sirnak",
    "srnak": "sirnak",
    "tekirdag": "tekirdag",
    "usak": "usak",
    "zinguldak": "zonguldak",
    "zonguldak": "zonguldak",
}


st.set_page_config(
    page_title="Turkiye Justice Risk Dashboard",
    layout="wide",
)


@st.cache_data
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [MASTER_PATH, MODELING_PATH, RAW_SGK_PATH, RAW_MEB_PATH]
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError(
            "Processed files are missing. Run `python src/prepare_raw_data.py` "
            "and `python src/merge_master_data.py` first."
        )

    master_df = pd.read_csv(MASTER_PATH)
    modeling_df = pd.read_csv(MODELING_PATH)
    raw_sgk_df = pd.read_csv(RAW_SGK_PATH)
    raw_meb_df = pd.read_csv(RAW_MEB_PATH)
    return master_df, modeling_df, raw_sgk_df, raw_meb_df


@st.cache_data
def load_geojson() -> dict:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"GeoJSON file not found at {GEOJSON_PATH}. "
            "Add the Turkey province boundary file first."
        )

    with GEOJSON_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_benchmark_artifacts() -> tuple[pd.DataFrame, dict[str, dict]] | tuple[None, None]:
    if not BENCHMARK_SUMMARY_PATH.exists() or not BENCHMARK_RESULTS_PATH.exists():
        return None, None

    benchmark_summary = pd.read_csv(BENCHMARK_SUMMARY_PATH)
    with BENCHMARK_RESULTS_PATH.open(encoding="utf-8") as file:
        benchmark_results = json.load(file)

    benchmark_lookup = {
        variant["title"]: {result["model_name"]: result for result in variant["model_results"]}
        for variant in benchmark_results
    }
    return benchmark_summary, benchmark_lookup


@st.cache_data
def load_model_predictions() -> pd.DataFrame | None:
    if not MODEL_PREDICTIONS_PATH.exists():
        return None

    prediction_df = pd.read_csv(MODEL_PREDICTIONS_PATH)
    prediction_df["year"] = pd.to_numeric(prediction_df["year"], errors="coerce")
    prediction_df["predicted_probability"] = pd.to_numeric(
        prediction_df["predicted_probability"], errors="coerce"
    )
    prediction_df["predicted_class"] = pd.to_numeric(
        prediction_df["predicted_class"], errors="coerce"
    )
    prediction_df["predicted_risk_label"] = prediction_df["predicted_class"].map(
        {1: "High", 0: "Low"}
    )
    return prediction_df


def format_number(value, digits: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def format_percent(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def format_score(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def prettify_model_name(value: str) -> str:
    mapping = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    return mapping.get(value, str(value).replace("_", " ").title())


def choose_default_variant(benchmark_summary_df: pd.DataFrame, model_name: str) -> str:
    model_rows = benchmark_summary_df.loc[benchmark_summary_df["model_name"] == model_name].copy()
    top_row = model_rows.sort_values(by=["cv_f1_mean", "test_f1_score"], ascending=False).iloc[0]
    return str(top_row["variant"])


def turkish_sort_key(value: str) -> list[int]:
    text = str(value).strip().lower()
    normalized = []
    for char in text:
        if char in TURKISH_ALPHABET:
            normalized.append(TURKISH_ALPHABET.index(char))
        else:
            normalized.append(len(TURKISH_ALPHABET) + ord(char))
    return normalized


def normalize_province_name(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return PROVINCE_NAME_ALIASES.get(text, text)


def describe_migration(value) -> str:
    if pd.isna(value):
        return "N/A"
    if value > 0:
        return f"{format_number(value)} in-migration"
    if value < 0:
        return f"{format_number(abs(value))} out-migration"
    return "Balanced"


def build_yearly_risk_labels(master_df: pd.DataFrame) -> pd.DataFrame:
    risk_df = master_df[["province", "year", "investigation_files_opened"]].dropna().copy()

    def assign_tertiles(group: pd.DataFrame) -> pd.DataFrame:
        ranked = group["investigation_files_opened"].rank(method="first")
        bins = pd.qcut(ranked, q=3, labels=["Low", "Medium", "High"])
        group = group.copy()
        group["risk_label"] = bins.astype(str)
        return group

    risk_df = risk_df.groupby("year", group_keys=False).apply(assign_tertiles)
    return risk_df[["province", "year", "risk_label"]]


def build_hover_chart(dataframe: pd.DataFrame, y_title: str, tooltip_format: str) -> alt.Chart:
    base = alt.Chart(dataframe).encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("Value:Q", title=y_title),
        color=alt.Color("Metric:N", title="Metric"),
    )

    lines = base.mark_line(strokeWidth=3)
    points = base.mark_circle(size=120).encode(
        tooltip=[
            alt.Tooltip("Year:O", title="Year"),
            alt.Tooltip("Metric:N", title="Metric"),
            alt.Tooltip("Value:Q", title="Value", format=tooltip_format),
        ]
    )

    return (lines + points).properties(height=280).interactive()


def build_geojson_for_year(geojson_data: dict, year_df: pd.DataFrame, selected_province_name: str) -> dict:
    year_lookup = {
        normalize_province_name(row["province"]): row
        for _, row in year_df.iterrows()
    }

    choropleth = deepcopy(geojson_data)
    for feature in choropleth["features"]:
        properties = feature.get("properties", {})
        province_name = properties.get("Name", "")
        province_key = normalize_province_name(province_name)
        row = year_lookup.get(province_key)

        risk_label = "No Data"
        files_opened = None
        predicted_probability = None
        fill_color = [90, 100, 115, 110]
        line_color = [230, 230, 230, 160]
        line_width = 1

        if row is not None:
            risk_label = row.get("dashboard_risk_label", row.get("risk_label", "No Data")) or "No Data"
            files_opened = row.get("investigation_files_opened")
            predicted_probability = row.get("predicted_probability")

            if risk_label == "High":
                fill_color = [232, 63, 63, 190]
            elif risk_label == "Medium":
                fill_color = [245, 158, 11, 185]
            elif risk_label == "Low":
                fill_color = [63, 131, 248, 180]

            if normalize_province_name(selected_province_name) == province_key:
                line_color = [255, 255, 255, 230]
                line_width = 4

        properties["province"] = province_name
        properties["risk_label"] = risk_label
        properties["investigation_files_opened"] = (
            int(files_opened) if pd.notna(files_opened) else None
        )
        properties["predicted_probability"] = (
            f"{predicted_probability:.3f}" if pd.notna(predicted_probability) else "N/A"
        )
        properties["fill_r"] = fill_color[0]
        properties["fill_g"] = fill_color[1]
        properties["fill_b"] = fill_color[2]
        properties["fill_a"] = fill_color[3]
        properties["line_r"] = line_color[0]
        properties["line_g"] = line_color[1]
        properties["line_b"] = line_color[2]
        properties["line_a"] = line_color[3]
        properties["line_width"] = line_width

    return choropleth


master_df, modeling_df, raw_sgk_df, raw_meb_df = load_datasets()
geojson_data = load_geojson()
benchmark_summary_df, benchmark_lookup = load_benchmark_artifacts()
model_predictions_df = load_model_predictions()

risk_labels = build_yearly_risk_labels(master_df)
app_df = master_df.merge(risk_labels, on=["province", "year"], how="left")
app_df = app_df.loc[~app_df["province"].astype(str).str.upper().str.contains("TÜRKİYE|TURKIYE")].copy()

years = sorted(app_df["year"].dropna().unique().tolist())
provinces = sorted(app_df["province"].dropna().unique().tolist(), key=turkish_sort_key)

if "selected_province" not in st.session_state:
    st.session_state["selected_province"] = provinces[0]
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = years[-1]

st.title("Turkiye Socio-Economic Justice Risk Dashboard")
st.caption(
    "Provincial dashboard built from SGK, migration, education, and justice proxy data."
)

with st.sidebar:
    st.header("Controls")
    selected_year = st.selectbox("Year", years, key="selected_year")
    selected_province = st.selectbox("Province", provinces, key="selected_province")
    if (
        benchmark_summary_df is not None
        and not benchmark_summary_df.empty
        and model_predictions_df is not None
        and not model_predictions_df.empty
    ):
        available_models = benchmark_summary_df["model_name"].dropna().unique().tolist()
        default_model_index = available_models.index("random_forest") if "random_forest" in available_models else 0
        selected_model_name = st.selectbox(
            "Benchmark Model",
            available_models,
            index=default_model_index,
            format_func=prettify_model_name,
        )
        selected_variant_name = st.selectbox(
            "Prediction Variant",
            benchmark_summary_df.loc[
                benchmark_summary_df["model_name"] == selected_model_name, "variant"
            ].tolist(),
            index=benchmark_summary_df.loc[
                benchmark_summary_df["model_name"] == selected_model_name, "variant"
            ].tolist().index(choose_default_variant(benchmark_summary_df, selected_model_name)),
        )
    else:
        selected_model_name = None
        selected_variant_name = None

if selected_model_name and selected_variant_name and model_predictions_df is not None:
    selected_predictions = model_predictions_df.loc[
        (model_predictions_df["model_name"] == selected_model_name)
        & (model_predictions_df["variant"] == selected_variant_name),
        ["province", "year", "predicted_class", "predicted_probability", "predicted_risk_label"],
    ].copy()
    app_display_df = app_df.merge(selected_predictions, on=["province", "year"], how="left")
    app_display_df["dashboard_risk_label"] = app_display_df["predicted_risk_label"].fillna(app_display_df["risk_label"])
else:
    app_display_df = app_df.copy()
    app_display_df["dashboard_risk_label"] = app_display_df["risk_label"]
    app_display_df["predicted_probability"] = pd.NA
    app_display_df["predicted_class"] = pd.NA
    app_display_df["predicted_risk_label"] = pd.NA

province_df = app_display_df.loc[app_display_df["province"] == selected_province].sort_values("year")
year_df = app_display_df.loc[app_display_df["year"] == selected_year].sort_values(
    ["predicted_probability", "investigation_files_opened"], ascending=[False, False]
)
selected_row = province_df.loc[province_df["year"] == selected_year]
selected_row = selected_row.iloc[0] if not selected_row.empty else None

recent_sgk_df = raw_sgk_df.loc[raw_sgk_df["province"] == selected_province].copy()
recent_meb_df = raw_meb_df.loc[raw_meb_df["province"] == selected_province].copy()
recent_sgk_df["year"] = pd.to_numeric(recent_sgk_df["year"], errors="coerce")
recent_meb_df["year"] = pd.to_numeric(recent_meb_df["year"], errors="coerce")
recent_sgk_df = recent_sgk_df.sort_values("year")
recent_meb_df = recent_meb_df.sort_values("year")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Investigation Files Opened",
    format_number(selected_row["investigation_files_opened"]) if selected_row is not None else "N/A",
)
col2.metric(
    "Active Insured Total",
    format_number(selected_row["active_insured_total"]) if selected_row is not None else "N/A",
)
col3.metric(
    "Migration Direction",
    describe_migration(selected_row["net_migration"]) if selected_row is not None else "N/A",
)
col4.metric(
    "Selected Model Risk" if selected_model_name else "Risk Label",
    selected_row["dashboard_risk_label"]
    if selected_row is not None and pd.notna(selected_row["dashboard_risk_label"])
    else "N/A",
)

col5, col6, col7 = st.columns(3)
col5.metric(
    "General Secondary Gross Enrollment",
    format_percent(selected_row["general_secondary_gross_enrollment_rate"])
    if selected_row is not None
    else "N/A",
)
col6.metric(
    "Vocational Secondary Gross Enrollment",
    format_percent(selected_row["vocational_secondary_gross_enrollment_rate"])
    if selected_row is not None
    else "N/A",
)
col7.metric(
    "Predicted High-Risk Probability",
    format_percent(selected_row["predicted_probability"] * 100, 1)
    if selected_row is not None and pd.notna(selected_row["predicted_probability"])
    else "N/A",
)

tab_labels = ["Main Risk View (2011-2021)", "Recent Trends View (2011-2025)"]
if benchmark_summary_df is not None and not benchmark_summary_df.empty:
    tab_labels.append("Model Benchmark View")
tabs = st.tabs(tab_labels)
main_tab = tabs[0]
recent_tab = tabs[1]
benchmark_tab = tabs[2] if len(tabs) > 2 else None

with main_tab:
    if selected_model_name and selected_variant_name:
        st.caption(
            f"Current model overlay: {prettify_model_name(selected_model_name)} "
            f"on {selected_variant_name}."
        )
    st.markdown("### Province Trend")
    volume_trend_df = province_df[
        ["year", "investigation_files_opened", "active_insured_total"]
    ].copy()
    volume_trend_df = volume_trend_df.rename(
        columns={
            "year": "Year",
            "investigation_files_opened": "Justice Files Opened",
            "active_insured_total": "Active Insured",
        }
    ).melt(id_vars="Year", var_name="Metric", value_name="Value")

    rate_trend_df = province_df[
        [
            "year",
            "general_secondary_gross_enrollment_rate",
            "vocational_secondary_gross_enrollment_rate",
            "university_rate",
            "predicted_probability",
        ]
    ].copy()
    rate_trend_df = rate_trend_df.rename(
        columns={
            "year": "Year",
            "general_secondary_gross_enrollment_rate": "General Secondary Gross Enrollment",
            "vocational_secondary_gross_enrollment_rate": "Vocational Secondary Gross Enrollment",
            "university_rate": "University Rate",
            "predicted_probability": "Predicted High-Risk Probability",
        }
    ).melt(id_vars="Year", var_name="Metric", value_name="Value")
    rate_trend_df = rate_trend_df.dropna(subset=["Value"]).copy()
    metric_counts = rate_trend_df.groupby("Metric")["Value"].count()
    valid_metrics = metric_counts[metric_counts > 1].index.tolist()
    rate_trend_df = rate_trend_df.loc[rate_trend_df["Metric"].isin(valid_metrics)].copy()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Volume Metrics**")
        st.altair_chart(
            build_hover_chart(volume_trend_df, "Count", ",.0f"),
            width="stretch",
        )

    with chart_col2:
        st.markdown("**Education And Model Signal**")
        if rate_trend_df.empty:
            st.info("No multi-year education or model trend is available for the selected province.")
        else:
            st.altair_chart(
                build_hover_chart(rate_trend_df, "Rate / Percent", ".2f"),
                width="stretch",
            )

with recent_tab:
    st.markdown("### Recent Monitoring Trend")
    st.caption(
        "This view extends beyond 2021 using SGK and MEB education series. "
        "It is for monitoring recent changes, not for justice-risk labeling."
    )
    recent_trend_df = recent_sgk_df[["year", "active_insured_total"]].rename(
        columns={"year": "Year", "active_insured_total": "Active Insured"}
    )
    recent_trend_df = recent_trend_df.merge(
        recent_meb_df[
            [
                "year",
                "general_secondary_gross_enrollment_rate",
                "vocational_secondary_gross_enrollment_rate",
            ]
        ].rename(
            columns={
                "year": "Year",
                "general_secondary_gross_enrollment_rate": "General Secondary Gross Enrollment",
                "vocational_secondary_gross_enrollment_rate": "Vocational Secondary Gross Enrollment",
            }
        ),
        on="Year",
        how="outer",
    ).sort_values("Year")

    recent_volume_df = recent_trend_df[["Year", "Active Insured"]].melt(
        id_vars="Year", var_name="Metric", value_name="Value"
    )
    recent_education_df = recent_trend_df[
        ["Year", "General Secondary Gross Enrollment", "Vocational Secondary Gross Enrollment"]
    ].melt(id_vars="Year", var_name="Metric", value_name="Value")
    recent_education_df = recent_education_df.dropna(subset=["Value"])

    recent_col1, recent_col2 = st.columns(2)
    with recent_col1:
        st.markdown("**Active Insured Trend**")
        st.altair_chart(
            build_hover_chart(recent_volume_df, "Count", ",.0f"),
            width="stretch",
        )
    with recent_col2:
        st.markdown("**MEB Gross Enrollment Trend**")
        st.altair_chart(
            build_hover_chart(recent_education_df, "Percent", ".2f"),
            width="stretch",
        )

if benchmark_tab is not None:
    with benchmark_tab:
        if selected_model_name is None:
            st.info("Run `python src/train.py` first to load benchmark model selections and saved predictions.")
        else:
            st.markdown("### Model Benchmark Comparison")
            st.caption(
                "Use the model selector in the sidebar to switch between benchmarked classifiers "
                "and compare their train/test stability across both modeling variants."
            )

            selected_model_rows = benchmark_summary_df.loc[
                benchmark_summary_df["model_name"] == selected_model_name
            ].copy()
            selected_model_rows["model_display_name"] = selected_model_rows["model_name"].map(prettify_model_name)

            top_variant = selected_model_rows.sort_values(
                by=["cv_f1_mean", "test_f1_score"],
                ascending=False,
            ).iloc[0]

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Selected Model", prettify_model_name(selected_model_name))
            metric_col2.metric("Best Variant", top_variant["variant"])
            metric_col3.metric("Top CV F1", format_score(top_variant["cv_f1_mean"]))
            metric_col4.metric("Top Test F1", format_score(top_variant["test_f1_score"]))

            display_df = selected_model_rows[
                [
                    "variant",
                    "rows_used",
                    "cv_folds",
                    "test_accuracy",
                    "test_f1_score",
                    "test_roc_auc",
                    "cv_accuracy_mean",
                    "cv_f1_mean",
                    "cv_roc_auc_mean",
                ]
            ].copy()
            display_df = display_df.rename(
                columns={
                    "variant": "Variant",
                    "rows_used": "Rows Used",
                    "cv_folds": "CV Folds",
                    "test_accuracy": "Test Accuracy",
                    "test_f1_score": "Test F1",
                    "test_roc_auc": "Test ROC-AUC",
                    "cv_accuracy_mean": "CV Accuracy",
                    "cv_f1_mean": "CV F1",
                    "cv_roc_auc_mean": "CV ROC-AUC",
                }
            )
            st.dataframe(display_df, width="stretch", hide_index=True)

            st.markdown("### Variant Details")
            selected_variant = st.selectbox(
                "Benchmark Variant",
                selected_model_rows["variant"].tolist(),
                index=selected_model_rows["variant"].tolist().index(selected_variant_name)
                if selected_variant_name in selected_model_rows["variant"].tolist()
                else 0,
                key="benchmark_variant",
            )
            model_result = benchmark_lookup[selected_variant][selected_model_name]
            test_metrics = model_result["test_metrics"]
            cv_metrics = model_result["cross_validation"]

            detail_col1, detail_col2, detail_col3 = st.columns(3)
            detail_col1.metric("Test Balanced Accuracy", format_score(test_metrics["balanced_accuracy"]))
            detail_col2.metric("Test Recall", format_score(test_metrics["recall"]))
            detail_col3.metric("CV Mean ROC-AUC", format_score(cv_metrics["roc_auc"]["mean"]))

            confusion_matrix_df = pd.DataFrame(
                model_result["test_metrics"]["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )

            detail_table = pd.DataFrame(
                {
                    "Metric": [
                        "Test Accuracy",
                        "Test Precision",
                        "Test Recall",
                        "Test F1",
                        "Test ROC-AUC",
                        "CV Accuracy Mean",
                        "CV F1 Mean",
                        "CV ROC-AUC Mean",
                    ],
                    "Value": [
                        test_metrics["accuracy"],
                        test_metrics["precision"],
                        test_metrics["recall"],
                        test_metrics["f1_score"],
                        test_metrics.get("roc_auc"),
                        cv_metrics["accuracy"]["mean"],
                        cv_metrics["f1"]["mean"],
                        cv_metrics["roc_auc"]["mean"],
                    ],
                }
            )
            detail_table["Value"] = detail_table["Value"].apply(format_score)

            benchmark_col1, benchmark_col2 = st.columns(2)
            with benchmark_col1:
                st.markdown("**Selected Model Metrics**")
                st.dataframe(detail_table, width="stretch", hide_index=True)
            with benchmark_col2:
                st.markdown("**Test Confusion Matrix**")
                st.dataframe(confusion_matrix_df, width="stretch")

st.markdown("### Turkey Province Map")
year_geojson = build_geojson_for_year(geojson_data, year_df, selected_province)
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=39.0,
        longitude=35.0,
        zoom=4.8,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "GeoJsonLayer",
            data=year_geojson,
            id="province-boundaries",
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            auto_highlight=False,
            get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a]",
            get_line_color="[properties.line_r, properties.line_g, properties.line_b, properties.line_a]",
            get_line_width="properties.line_width",
            line_width_min_pixels=1,
        )
    ],
    tooltip={
        "html": "<b>{province}</b><br/>Justice Files Opened: {investigation_files_opened}<br/>Displayed Risk: {risk_label}<br/>Predicted High-Risk Probability: {predicted_probability}",
        "style": {"backgroundColor": "#111827", "color": "white"},
    },
)
st.pydeck_chart(deck, width="stretch")

st.markdown("### Selected Province Snapshot")
snapshot_cols = [
    "province",
    "year",
    "geographical_region",
    "statistical_region",
    "population",
    "in_migration",
    "out_migration",
    "net_migration",
    "active_insured_total",
    "upper_secondary_gross_enrollment_rate",
    "general_secondary_gross_enrollment_rate",
    "vocational_secondary_gross_enrollment_rate",
    "illiterate_rate",
    "upper_secondary_rate",
    "university_rate",
    "higher_education_share",
    "investigation_files_total_load",
    "investigation_files_opened",
    "investigation_files_closed",
    "risk_label",
    "dashboard_risk_label",
    "predicted_probability",
]
if selected_row is not None:
    st.dataframe(
        pd.DataFrame([selected_row[snapshot_cols]]).reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No row found for the selected province and year.")

ranking_header_col1, ranking_header_col2 = st.columns([4, 1])
with ranking_header_col1:
    st.markdown("### Province Ranking For Selected Year")
with ranking_header_col2:
    ranking_limit = st.selectbox(
        "Province Ranking Size",
        [15, 30, 50, 81],
        index=0,
        label_visibility="collapsed",
        key="ranking_limit",
    )

ranking_df = year_df[
    [
        "province",
        "predicted_probability",
        "dashboard_risk_label",
        "investigation_files_opened",
        "active_insured_total",
        "net_migration",
    ]
].head(ranking_limit)
ranking_df["predicted_probability"] = ranking_df["predicted_probability"].apply(
    lambda value: format_percent(value * 100, 1) if pd.notna(value) else "N/A"
)
ranking_df = ranking_df.rename(
    columns={
        "province": "Province",
        "predicted_probability": "Predicted High-Risk Probability",
        "dashboard_risk_label": "Displayed Risk",
        "investigation_files_opened": "Justice Files Opened",
        "active_insured_total": "Active Insured",
        "net_migration": "Net Migration",
    }
)
st.dataframe(ranking_df.reset_index(drop=True), width="stretch", hide_index=True)

st.markdown("### Modeling Coverage")
coverage_col1, coverage_col2, coverage_col3 = st.columns(3)
coverage_col1.metric("Master Rows", format_number(len(master_df)))
coverage_col2.metric("Modeling Rows", format_number(len(modeling_df)))
coverage_col3.metric("Province Count", format_number(app_df["province"].nunique()))

st.info(
    "When a benchmark model is selected, displayed risk labels and map colors use the saved model "
    "predictions for the chosen variant. Justice-file metrics still reflect the observed dataset, "
    "and predicted labels remain a proxy rather than absolute real-world crime truth."
)
