
import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import load_data, calculate_metrics, save_outputs

st.set_page_config(
    page_title="Employee Experience Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Engagement, Satisfaction & Burnout Diagnostics")
st.caption(
    "Palo Alto Networks | HR Analytics Project | Unified Mentor"
)

# Load and analyze the data.
@st.cache_data
def get_data():
    raw = load_data()
    return calculate_metrics(raw)


try:
    df = get_data()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.info(
        "Check that the CSV is in the data folder and that its "
        "column names match the project requirements."
    )
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

departments = sorted(df["Department"].dropna().unique().tolist())
selected_departments = st.sidebar.multiselect(
    "Department",
    departments,
    default=departments,
)

roles = sorted(df["JobRole"].dropna().unique().tolist())
selected_roles = st.sidebar.multiselect(
    "Job Role",
    roles,
    default=roles,
)

overtime_options = ["All"] + sorted(
    df["OverTime"].dropna().unique().tolist()
)
selected_overtime = st.sidebar.selectbox(
    "Overtime",
    overtime_options,
)

job_levels = sorted(
    df["JobLevel"].dropna().unique().tolist()
)
selected_levels = st.sidebar.multiselect(
    "Job Level",
    job_levels,
    default=job_levels,
)

max_tenure = int(df["YearsAtCompany"].max())
tenure_range = st.sidebar.slider(
    "Years at Company",
    min_value=0,
    max_value=max(1, max_tenure),
    value=(0, max(1, max_tenure)),
)

engagement_threshold = st.sidebar.slider(
    "Low Engagement Alert Threshold",
    min_value=0,
    max_value=100,
    value=60,
)

# Apply all filters.
filtered = df[
    df["Department"].isin(selected_departments)
    & df["JobRole"].isin(selected_roles)
    & df["JobLevel"].isin(selected_levels)
    & df["YearsAtCompany"].between(
        tenure_range[0], tenure_range[1]
    )
].copy()

if selected_overtime != "All":
    filtered = filtered[
        filtered["OverTime"] == selected_overtime
    ]

if filtered.empty:
    st.warning("No employees match these filters.")
    st.stop()

# ---------------- KPI CARDS ----------------
def metric_mean(series):
    value = series.mean()
    return f"{value:.1f}%" if pd.notna(value) else "N/A"


total = len(filtered)
avg_engagement = metric_mean(filtered["EngagementScore"])
low_engagement_pct = (
    filtered["EngagementScore"].lt(engagement_threshold).mean() * 100
)
high_risk_count = filtered["BurnoutRiskLevel"].eq("High").sum()
avg_balance = metric_mean(filtered["WorkLifeBalanceIndex"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees in View", f"{total:,}")
c2.metric("Average Engagement", avg_engagement)
c3.metric("Below Threshold", f"{low_engagement_pct:.1f}%")
c4.metric("High-Risk Indicators", f"{high_risk_count:,}")

st.caption(
    "Scores are descriptive indicators calculated from the available "
    "dataset. They are not validated clinical or predictive assessments."
)

# ---------------- DASHBOARD TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Engagement Health",
    "Burnout Risk",
    "Career & Role",
    "Manager Action Panel",
])

with tab1:
    st.subheader("Engagement Health Overview")

    col1, col2 = st.columns(2)

    with col1:
        engagement_data = filtered.dropna(
            subset=["EngagementScore"]
        )
        fig = px.histogram(
            engagement_data,
            x="EngagementScore",
            nbins=20,
            title="Employee Engagement Score Distribution",
            labels={"EngagementScore": "Engagement Score (0–100)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        satisfaction_columns = [
            "JobInvolvement",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
        ]
        satisfaction = (
            filtered[satisfaction_columns]
            .mean()
            .rename_axis("Dimension")
            .reset_index(name="Average Rating")
        )
        fig = px.bar(
            satisfaction,
            x="Dimension",
            y="Average Rating",
            title="Average Satisfaction Dimensions",
            range_y=[0, 4],
        )
        st.plotly_chart(fig, use_container_width=True)

    department_summary = (
        filtered.groupby("Department", observed=True)
        .agg(
            Employees=("Department", "size"),
            Engagement=("EngagementScore", "mean"),
            WorkLifeBalance=("WorkLifeBalanceIndex", "mean"),
        )
        .reset_index()
    )

    fig = px.bar(
        department_summary,
        x="Department",
        y="Engagement",
        color="Department",
        title="Average Engagement by Department",
        range_y=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Burnout Risk Dashboard")

    risk_order = ["Low", "Medium", "High"]
    risk_counts = (
        filtered["BurnoutRiskLevel"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk Level")
        .reset_index(name="Employees")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Employees",
            title="Diagnostic Risk Indicator Distribution",
            category_orders={"Risk Level": risk_order},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            x="OverTime",
            y="WorkLifeBalance",
            title="Work-Life Balance by Overtime",
            points="all",
            category_orders={
                "OverTime": ["No", "Yes"]
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    risk_by_department = (
        filtered.groupby("Department", observed=True)
        .agg(
            Employees=("Department", "size"),
            HighRisk=("BurnoutRiskLevel",
                      lambda s: s.eq("High").sum()),
        )
        .reset_index()
    )
    risk_by_department["High Risk %"] = (
        risk_by_department["HighRisk"]
        / risk_by_department["Employees"] * 100
    )

    fig = px.bar(
        risk_by_department,
        x="Department",
        y="High Risk %",
        title="High-Risk Indicator Percentage by Department",
        hover_data=["Employees", "HighRisk"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "High risk here means overtime plus a work-life balance "
        "rating of 1 or 2. It does not establish actual burnout."
    )

with tab3:
    st.subheader("Role & Career Stage Analysis")

    role_summary = (
        filtered.groupby("JobRole", observed=True)
        .agg(
            Employees=("JobRole", "size"),
            Engagement=("EngagementScore", "mean"),
        )
        .reset_index()
        .sort_values("Engagement")
    )

    fig = px.bar(
        role_summary,
        x="Engagement",
        y="JobRole",
        orientation="h",
        title="Average Engagement by Job Role",
        hover_data=["Employees"],
        range_x=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)

    level_summary = (
        filtered.groupby("JobLevel", observed=True)
        .agg(
            Employees=("JobLevel", "size"),
            Engagement=("EngagementScore", "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            level_summary,
            x="JobLevel",
            y="Engagement",
            markers=True,
            title="Engagement by Job Level",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tenure_summary = (
            filtered.groupby("YearsAtCompany", observed=True)
            ["EngagementScore"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            tenure_summary,
            x="YearsAtCompany",
            y="EngagementScore",
            markers=True,
            title="Engagement vs Company Tenure",
        )
        st.plotly_chart(fig, use_container_width=True)

    promotion_summary = (
        filtered.groupby("YearsSinceLastPromotion", observed=True)
        ["EngagementScore"]
        .mean()
        .reset_index()
    )
    fig = px.scatter(
        promotion_summary,
        x="YearsSinceLastPromotion",
        y="EngagementScore",
        title="Promotion Gap vs Average Engagement",
        labels={
            "YearsSinceLastPromotion": "Years Since Last Promotion",
            "EngagementScore": "Average Engagement Score",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Manager Action Panel")

    alerts = filtered[
        filtered["EngagementScore"].lt(engagement_threshold)
        | filtered["BurnoutRiskLevel"].eq("High")
    ].copy()

    st.metric("Employees Flagged for Review", len(alerts))

    if alerts.empty:
        st.success(
            "No employees meet the current diagnostic alert criteria."
        )
    else:
        st.warning(
            "Review these indicators confidentially. Use conversations "
            "and employee feedback before making decisions."
        )

        alert_columns = [
            "Department",
            "JobRole",
            "JobLevel",
            "YearsAtCompany",
            "OverTime",
            "WorkLifeBalance",
            "EngagementScore",
            "BurnoutRiskLevel",
        ]
        st.dataframe(
            alerts[alert_columns].sort_values(
                "EngagementScore", na_position="first"
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Department-level priorities")

    priorities = (
        filtered.groupby("Department", observed=True)
        .agg(
            Employees=("Department", "size"),
            Engagement=("EngagementScore", "mean"),
            HighRisk=("BurnoutRiskLevel",
                      lambda s: s.eq("High").sum()),
        )
        .reset_index()
    )

    priorities["Below Threshold %"] = (
        filtered.assign(
            Below=filtered["EngagementScore"].lt(
                engagement_threshold
            )
        )
        .groupby("Department", observed=True)["Below"]
        .mean()
        .reindex(priorities["Department"])
        .to_numpy() * 100
    )

    priorities = priorities.sort_values(
        ["Below Threshold %", "HighRisk"],
        ascending=False,
    )

    st.dataframe(
        priorities.round(2),
        use_container_width=True,
        hide_index=True,
    )

# ---------------- DOWNLOAD REPORT ----------------
st.divider()
st.subheader("Export Analysis")

if st.button("Generate and save analysis report"):
    save_outputs(df)
    st.success(
        "Full calculated dataset saved to outputs/analysis_report.csv"
    )

st.download_button(
    "Download filtered employee report",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="employee_engagement_filtered_report.csv",
    mime="text/csv",
)

st.caption(
    "Internal project analysis. Treat employee-level information as "
    "confidential and restrict access to authorized reviewers."
)
