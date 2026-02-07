import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)

# ---------------- LOAD MODEL ----------------
def load_model():
    try:
        return pickle.load(open("student_model.pkl", "rb"))
    except Exception:
        return None

# ---------------- LOAD METRICS ----------------
def load_metrics():
    try:
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    except Exception:
        return None

if "model" not in st.session_state:
    st.session_state["model"] = load_model()

if "metrics" not in st.session_state:
    st.session_state["metrics"] = load_metrics()

if "training_attempted" not in st.session_state:
    st.session_state["training_attempted"] = False

if (st.session_state["model"] is None or st.session_state["metrics"] is None) and not st.session_state["training_attempted"]:
    st.session_state["training_attempted"] = True
    with st.spinner("Training model from dataset..."):
        try:
            from train_model import train_and_save

            train_and_save()
            st.session_state["model"] = load_model()
            st.session_state["metrics"] = load_metrics()
        except Exception:
            pass

model = st.session_state["model"]
metrics = st.session_state["metrics"]

if model is None:
    st.warning("⚠ Model file not found. Ensure training completes successfully.")

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Predictor Page", "Admin Page"], index=0)

#ADMIN PAGE
if page == "Admin Page":

    st.caption("Model performance, data preview, and insights.")

    if metrics:
        st.subheader("Model Performance")

        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        st.metric("F1", f"{metrics.get('f1', 0):.3f}")
        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")

        st.caption(f"Decision threshold: {metrics.get('best_threshold', 0.5):.2f}")

        cm = metrics.get("confusion_matrix")
        if cm:
            st.subheader("Confusion Matrix")
            st.table(
                pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
            )

        model_results = metrics.get("model_results")
        if model_results:
            st.subheader("Model-wise Outputs")
            model_df = pd.DataFrame(model_results)
            st.dataframe(model_df, use_container_width=True)

    else:
        st.info("Train the model to see live metrics.")

    # -------- Data Preview --------
    st.subheader("Data Preview")
    try:
        head_df = pd.read_csv("student_data_head.csv")
        st.dataframe(head_df, use_container_width=True, height=240)
    except Exception:
        st.caption("No preview available.")

    # -------- Heatmap --------
    st.subheader("Pass/Fail Heatmap")
    try:
        heat_df = pd.read_csv("student_data_cleaned.csv")

        if metrics and metrics.get("feature_names"):
            feature_names = metrics["feature_names"]
            cols = [c for c in feature_names if c in heat_df.columns]
            if "Pass" in heat_df.columns:
                cols.append("Pass")
            heat_df = heat_df[cols]

        corr = heat_df.corr(numeric_only=True)
        corr_reset = corr.reset_index().melt(
            id_vars="index",
            var_name="Feature",
            value_name="Correlation",
        )

        st.vega_lite_chart(
            corr_reset,
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "Feature", "type": "nominal"},
                    "y": {"field": "index", "type": "nominal"},
                    "color": {
                        "field": "Correlation",
                        "type": "quantitative",
                        "scale": {"scheme": "redblue"},
                    },
                    "tooltip": [
                        {"field": "index", "type": "nominal"},
                        {"field": "Feature", "type": "nominal"},
                        {"field": "Correlation", "type": "quantitative", "format": ".2f"},
                    ],
                },
            },
            use_container_width=True,
        )

    except Exception:
        st.caption("Heatmap unavailable.")

    # -------- Pass Rate Relations --------
    st.subheader("Pass Rate Relations")
    try:
        reg_df = pd.read_csv("student_data_cleaned.csv")

        if "Pass" in reg_df.columns:
            x_field = "TotalStudyHours" if "TotalStudyHours" in reg_df.columns else None
            if not x_field and "TotalAttendance" in reg_df.columns:
                x_field = "TotalAttendance"

            if x_field:
                reg_df = reg_df.dropna()

                candidate_fields = [
                    "TotalStudyHours",
                    "TotalAttendance",
                    "PreviousGrade",
                    "ExtracurricularActivities",
                ]

                feature_fields = [
                    field for field in candidate_fields
                    if field in reg_df.columns and field != "Pass"
                ]

                if not feature_fields:
                    feature_fields = [x_field]

                cols = st.columns(2)
                for idx, field in enumerate(feature_fields):
                    chart_df = reg_df[[field, "Pass"]].dropna()
                    if chart_df.empty:
                        continue

                    chart_spec = {
                        "title": f"Pass rate by {field}",
                        "layer": [
                            {
                                "transform": [
                                    {"bin": {"maxbins": 20}, "field": field, "as": "x_bin"},
                                    {"aggregate": [
                                        {"op": "mean", "field": "Pass", "as": "pass_rate"},
                                        {"op": "count", "field": "Pass", "as": "count"},
                                    ], "groupby": ["x_bin"]},
                                ],
                                "mark": {"type": "line", "color": "#ff4b4b", "size": 2},
                                "encoding": {
                                    "x": {"field": "x_bin", "type": "quantitative", "title": field},
                                    "y": {
                                        "field": "pass_rate",
                                        "type": "quantitative",
                                        "title": "Pass Rate",
                                        "scale": {"domain": [0, 1]},
                                    },
                                },
                            },
                            {
                                "transform": [
                                    {"bin": {"maxbins": 20}, "field": field, "as": "x_bin"},
                                    {"aggregate": [
                                        {"op": "mean", "field": "Pass", "as": "pass_rate"},
                                        {"op": "count", "field": "Pass", "as": "count"},
                                    ], "groupby": ["x_bin"]},
                                ],
                                "mark": {"type": "circle", "color": "#ffd166", "size": 70},
                                "encoding": {
                                    "x": {"field": "x_bin", "type": "quantitative", "title": field},
                                    "y": {
                                        "field": "pass_rate",
                                        "type": "quantitative",
                                        "title": "Pass Rate",
                                        "scale": {"domain": [0, 1]},
                                    },
                                    "tooltip": [
                                        {"field": "x_bin", "type": "quantitative", "title": field},
                                        {"field": "pass_rate", "type": "quantitative", "format": ".2f", "title": "Pass Rate"},
                                        {"field": "count", "type": "quantitative", "title": "Samples"},
                                    ],
                                },
                            },
                        ],
                    }

                    with cols[idx % 2]:
                        st.vega_lite_chart(
                            chart_df,
                            chart_spec,
                            use_container_width=True,
                        )
            else:
                st.caption("Pass rate charts unavailable (missing study/attendance columns).")
        else:
            st.caption("Regression plot unavailable (missing Pass column).")
    except Exception:
        st.caption("Regression plot unavailable.")

# PREDICTOR PANEL 
else:

    st.caption("Provide student details to estimate pass likelihood.")

    gender_options = {"Female": 0, "Male": 1}
    parental_support_options = {"Low": 0, "Medium": 1, "High": 2}
    online_classes_options = {"No": 0, "Yes": 1}

    extracurricular_labels = {
        0: "None",
        1: "Some",
        2: "Active",
        3: "Very Active",
    }

    # -------- Inputs --------
    col1, col2 = st.columns(2)

    with col1:
        gender_label = st.selectbox("Gender", list(gender_options.keys()))
        parental_support_label = st.selectbox(
            "Parental Support",
            list(parental_support_options.keys()),
        )
        online_classes_label = st.selectbox(
            "Online Classes Taken",
            list(online_classes_options.keys()),
        )

    with col2:
        previous_grade = st.slider("Previous Grade", 0, 100, 75)
        extracurricular_level = st.slider("Extracurricular Activities", 0, 3, 1)
        st.caption(f"Selected: {extracurricular_labels[extracurricular_level]}")

    total_study = st.number_input("Total Study Hours of a week", 0.0, 50.0, 15.0)
    total_attendance = st.number_input("Attendance Score", 0.0, 100.0, 75.0)

    # -------- Prediction Map --------
    if model:
        with st.expander("Pass/Fail Map", expanded=True):

            grid_study = np.linspace(0, 50, 26)
            grid_attendance = np.linspace(0, 100, 26)

            rows = []
            for s in grid_study:
                for a in grid_attendance:
                    rows.append({
                        "Gender": gender_options[gender_label],
                        "PreviousGrade": previous_grade,
                        "ExtracurricularActivities": extracurricular_level,
                        "ParentalSupport": parental_support_options[parental_support_label],
                        "Online Classes Taken": online_classes_options[online_classes_label],
                        "TotalStudyHours": s,
                        "TotalAttendance": a,
                    })

            feature_names = (
                metrics.get("feature_names")
                if metrics and metrics.get("feature_names")
                else list(rows[0].keys())
            )

            grid_df = pd.DataFrame(rows, columns=feature_names)

            proba = model.predict_proba(grid_df)[:, 1]
            threshold = metrics.get("best_threshold", 0.5) if metrics else 0.5
            pred = (proba >= threshold).astype(int)

            chart_df = pd.DataFrame({
                "Study": np.repeat(grid_study, len(grid_attendance)),
                "Attendance": np.tile(grid_attendance, len(grid_study)),
                "Prediction": np.where(pred == 1, "Pass", "Fail"),
                "Probability": proba,
            })

            st.vega_lite_chart(
                chart_df,
                {
                    "mark": {"type": "circle", "size": 40},
                    "encoding": {
                        "x": {"field": "Study", "type": "quantitative"},
                        "y": {"field": "Attendance", "type": "quantitative"},
                        "color": {"field": "Prediction", "type": "nominal"},
                        "tooltip": [
                            {"field": "Study"},
                            {"field": "Attendance"},
                            {"field": "Prediction"},
                            {"field": "Probability", "format": ".2%"},
                        ],
                    },
                },
                use_container_width=True,
            )

    # -------- Predict Button --------
    if st.button("Predict", type="primary") and model:

        feature_row = {
            "Gender": gender_options[gender_label],
            "PreviousGrade": previous_grade,
            "ExtracurricularActivities": extracurricular_level,
            "ParentalSupport": parental_support_options[parental_support_label],
            "Online Classes Taken": online_classes_options[online_classes_label],
            "TotalStudyHours": total_study,
            "TotalAttendance": total_attendance,
        }

        feature_names = (
            metrics.get("feature_names")
            if metrics and metrics.get("feature_names")
            else list(feature_row.keys())
        )

        df = pd.DataFrame([feature_row], columns=feature_names)

        proba = model.predict_proba(df)[0][1]
        threshold = metrics.get("best_threshold", 0.5) if metrics else 0.5

        st.caption(f"Pass likelihood: {proba:.2%}")

        if proba >= threshold:
            st.success("Student is likely to PASS ✅")
        else:
            st.error("Student is likely to FAIL ❌")

    st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 8px;
}
</style>

<div class="footer">
© 2026 Student Performance Prediction App | Made with vision by Rishabh Barnwal & Rudraksha Sharma
</div>
""", unsafe_allow_html=True)
