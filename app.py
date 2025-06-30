import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("EA.csv")
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Data")
departments = st.sidebar.multiselect("Department", df["Department"].unique(), default=list(df["Department"].unique()))
genders = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
income_min, income_max = int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())
income_range = st.sidebar.slider("Monthly Income Range", income_min, income_max, (income_min, income_max))

# Filter Data
df_filtered = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["MonthlyIncome"].between(income_range[0], income_range[1]))
]

# Main Title
st.title("📊 Employee Attrition Insights Dashboard")
st.markdown("This dashboard enables HR Directors and stakeholders to deeply explore attrition and workforce trends using 20+ interactive charts, filters, and tables.")

# KPI Section
st.markdown("### 🔑 Key Metrics (for current filters)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", len(df_filtered))
attrition_rate = df_filtered["Attrition"].value_counts(normalize=True).get("Yes", 0)
col2.metric("Attrition Rate (%)", f"{attrition_rate * 100:.1f}%")
col3.metric("Avg Age", f"{df_filtered['Age'].mean():.1f}")
col4.metric("Avg Monthly Income", f"{df_filtered['MonthlyIncome'].mean():,.0f}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Attrition", "Demographics", "Compensation", "Performance & Satisfaction"
])

# -------- TAB 1: OVERVIEW --------
with tab1:
    st.header("🔍 Dataset Overview & Macro Analysis")
    st.write("Get a snapshot of your HR data and trends at a glance.")

    st.markdown("**Preview of Filtered Data**")
    st.dataframe(df_filtered.head(20), use_container_width=True)

    st.markdown("**1. Department-wise Employee Count**")
    st.write("Distribution of employees across departments shows workforce structure.")
    fig = px.histogram(df_filtered, x="Department", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**2. Correlation Heatmap**")
    st.write("Correlation between numeric features to spot strong drivers.")
    corr = df_filtered.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("**3. Attrition Count**")
    st.write("See overall attrition balance (Yes/No).")
    fig = px.histogram(df_filtered, x="Attrition", color="Attrition")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**4. Attrition Rate by Department**")
    st.write("Which departments lose more employees?")
    group = df_filtered.groupby("Department")["Attrition"].value_counts(normalize=True).rename("Rate").reset_index()
    group = group[group["Attrition"]=="Yes"]
    fig = px.bar(group, x="Department", y="Rate", text="Rate", labels={"Rate": "Attrition Rate"})
    st.plotly_chart(fig, use_container_width=True)

# -------- TAB 2: ATTRITION --------
with tab2:
    st.header("📉 Attrition Deep Dive")
    st.write("Explore employee attrition by multiple business variables.")

    st.markdown("**5. Attrition by Age Group**")
    st.write("Identifies at-risk age cohorts.")
    age_bins = pd.cut(df_filtered["Age"], bins=[18, 25, 35, 45, 55, 70])
    age_attr = pd.crosstab(age_bins, df_filtered["Attrition"], normalize='index')
    fig = px.bar(age_attr, barmode="group", title="Attrition Rate by Age Group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**6. Attrition by Education Field**")
    st.write("Reveals which education fields see more churn.")
    fig = px.histogram(df_filtered, x="EducationField", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**7. Attrition by Business Travel**")
    st.write("Does frequent travel cause attrition?")
    fig = px.histogram(df_filtered, x="BusinessTravel", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**8. Attrition by Job Role**")
    st.write("Find out which job roles are most/least stable.")
    fig = px.histogram(df_filtered, x="JobRole", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**9. Years at Company vs Attrition**")
    st.write("Are newcomers or veterans leaving more?")
    fig = px.histogram(df_filtered, x="YearsAtCompany", color="Attrition", nbins=15)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**10. Distance from Home vs Attrition**")
    st.write("Long commutes and attrition rates.")
    fig = px.box(df_filtered, x="Attrition", y="DistanceFromHome")
    st.plotly_chart(fig, use_container_width=True)

# -------- TAB 3: DEMOGRAPHICS --------
with tab3:
    st.header("👥 Demographics & Diversity")
    st.write("Analyze the workforce by gender, marital status, and diversity measures.")

    st.markdown("**11. Gender Distribution**")
    st.write("Male/Female distribution across organization.")
    fig = px.pie(df_filtered, names="Gender", title="Gender Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**12. Marital Status Distribution**")
    st.write("Workforce by marital status.")
    fig = px.pie(df_filtered, names="MaritalStatus", title="Marital Status")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**13. Attrition by Gender**")
    st.write("Does gender impact attrition?")
    gender_attr = df_filtered.groupby("Gender")["Attrition"].value_counts(normalize=True).rename("Rate").reset_index()
    gender_attr = gender_attr[gender_attr["Attrition"]=="Yes"]
    fig = px.bar(gender_attr, x="Gender", y="Rate", text="Rate", labels={"Rate": "Attrition Rate"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**14. Attrition by Marital Status**")
    st.write("Attrition by marital status.")
    fig = px.histogram(df_filtered, x="MaritalStatus", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**15. Age Distribution**")
    st.write("Age spread of the workforce.")
    fig = px.histogram(df_filtered, x="Age", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

# -------- TAB 4: COMPENSATION --------
with tab4:
    st.header("💵 Compensation Analysis")
    st.write("Drill down on salary, incentives, and benefits.")

    st.markdown("**16. Monthly Income by Job Role**")
    st.write("Salary distribution by job role.")
    fig = px.box(df_filtered, x="JobRole", y="MonthlyIncome", color="JobRole")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**17. Monthly Income by Attrition**")
    st.write("Income distribution among stayers/leavers.")
    fig = px.box(df_filtered, x="Attrition", y="MonthlyIncome", color="Attrition")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**18. Overtime vs Attrition**")
    st.write("Does overtime work link to attrition?")
    fig = px.histogram(df_filtered, x="OverTime", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**19. Percent Salary Hike by Attrition**")
    st.write("Salary growth vs attrition.")
    fig = px.box(df_filtered, x="Attrition", y="PercentSalaryHike", color="Attrition")
    st.plotly_chart(fig, use_container_width=True)

# -------- TAB 5: PERFORMANCE & SATISFACTION --------
with tab5:
    st.header("📈 Performance, Engagement & Satisfaction")
    st.write("Find patterns in ratings, satisfaction, and promotion.")

    st.markdown("**20. Job Satisfaction by Attrition**")
    st.write("Are unsatisfied employees leaving more?")
    fig = px.box(df_filtered, x="Attrition", y="JobSatisfaction", color="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**21. Environment Satisfaction by Attrition**")
    st.write("Satisfaction with work environment and attrition.")
    fig = px.box(df_filtered, x="Attrition", y="EnvironmentSatisfaction", color="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**22. Performance Rating Distribution**")
    st.write("Performance rating across the workforce.")
    fig = px.histogram(df_filtered, x="PerformanceRating", color="Attrition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**23. Work Life Balance by Attrition**")
    st.write("Correlation between work-life balance and attrition.")
    fig = px.box(df_filtered, x="Attrition", y="WorkLifeBalance", color="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**24. Years Since Last Promotion vs Attrition**")
    st.write("Do employees who are not promoted leave more?")
    fig = px.box(df_filtered, x="Attrition", y="YearsSinceLastPromotion", color="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**25. Years With Current Manager by Attrition**")
    st.write("Relationship with manager and attrition.")
    fig = px.box(df_filtered, x="Attrition", y="YearsWithCurrManager", color="Attrition", points="all")
    st.plotly_chart(fig, use_container_width=True)

st.info("All charts above are interactive. Adjust the filters in the sidebar for micro and macro insights!")
