import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components

from io import BytesIO
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Online Banking Cybersecurity Awareness Toolkit",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("Online_Banking_Cleaned_Dataset (1).csv")

df = load_data()

st.sidebar.title("Navigation")

pages = [
    "Home",
    "Dataset Overview",
    "Exploratory Data Analysis",
    "Cyber Awareness Assessment",
    "Biometric Training Demo",
    "Learning Resources"
]

page = st.sidebar.radio("Go to", pages)

if page == "Home":

    st.title("Online Banking Cybersecurity Awareness Toolkit")

    st.markdown("""
    ### Evaluating the Effectiveness of User Awareness and Front-End Security Measures in Online Banking
    """)

    st.write("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.write("""
        Welcome to the Online Banking Cybersecurity Awareness Toolkit.

        This application was developed as part of a Final Year Research Project to
        evaluate users' cybersecurity awareness and the effectiveness of front-end
        security measures used in online banking.

        The toolkit combines survey findings, exploratory data analysis,
        cybersecurity awareness assessment, personalised recommendations,
        and an educational biometric authentication demonstration.
        """)

    with col2:
        st.metric("Survey Responses", len(df))
        st.metric("Questionnaire Items", len(df.columns))
        st.metric("Toolkit Modules", len(pages))

    st.write("---")

    st.subheader("Toolkit Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("Cyber Awareness Assessment")

        st.write("""
        • Assess users' cybersecurity awareness

        • Calculate awareness percentage

        • Provide personalised recommendations
        """)

        st.success("Exploratory Data Analysis")

        st.write("""
        • Interactive charts

        • Dataset overview

        • Correlation analysis
        """)

    with col2:

        st.success("Biometric Training Demonstration")

        st.write("""
        • Face detection explanation

        • Facial landmark identification

        • Faceprint generation concept
        """)

        st.success("Learning Resources")

        st.write("""
        • Phishing awareness

        • Password security

        • Multi-Factor Authentication

        • AI-enabled cyber threats
        """)

    st.write("---")

    with st.expander("Project Aim"):

        st.write("""
        To evaluate the effectiveness of user awareness and front-end security
        measures in online banking and to develop an interactive educational
        toolkit that improves users' cybersecurity knowledge and promotes safer
        online banking practices.
        """)

    with st.expander("How to Use This Toolkit"):

        st.write("""
        1. Review the dataset overview.

        2. Explore the visualisations.

        3. Complete the Cyber Awareness Assessment.

        4. Read your personalised recommendations.

        5. Learn how biometric authentication works through the training demonstration.
        """)


elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.write("""
    This dataset was collected through an online questionnaire designed to
    evaluate users' cybersecurity awareness and their perceptions of front-end
    security measures in online banking.

    It includes demographic information, online banking usage, cybersecurity
    awareness, phishing experiences, practical security scenarios, security
    behaviour, opinions about authentication controls and biometric awareness.
    """)

    st.write("---")

    
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Responses", df.shape[0])
    col2.metric("Total Variables", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Records", int(df.duplicated().sum()))

    st.write("---")

    

    st.subheader("Dataset Information")

    dataset_info = pd.DataFrame({
        "Property": [
            "Number of Rows",
            "Number of Columns",
            "Missing Values",
            "Duplicate Records",
            "Memory Usage"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum()),
            f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    st.subheader("Survey Structure")

    survey_structure = pd.DataFrame({
        "Survey Section": [
            "Demographic Information",
            "Cybersecurity Awareness",
            "Cybersecurity Experience",
            "Practical Security Scenarios",
            "Opinions on Security Measures",
            "Online Banking Behaviour",
            "Biometric Authentication",
            "Other Variables"
        ],
        "Number of Variables": [
            7,
            7,
            6,
            4,
            5,
            4,
            6,
            2
        ]
    })

    st.dataframe(
        survey_structure,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Select the number of rows to display",
        min_value=5,
        max_value=min(20, len(df)),
        value=5
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True
    )

    st.write("---")

    st.subheader("Data Types")

    datatype_table = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        datatype_table,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")


    st.subheader("Data Type Summary")

    datatype_summary = (
        df.dtypes.astype(str)
        .value_counts()
        .reset_index()
    )

    datatype_summary.columns = [
        "Data Type",
        "Number of Variables"
    ]

    st.dataframe(
        datatype_summary,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")


    st.subheader("Missing Value Analysis")

    missing_table = pd.DataFrame({
        "Variable": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    if df.isnull().sum().sum() == 0:
        st.success(
            "The cleaned dataset contains no missing values."
        )

    st.write("---")

    st.subheader("Numerical Variable Summary")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numerical_columns) > 0:

        numerical_summary = (
            df[numerical_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        numerical_summary = numerical_summary.rename(
            columns={"index": "Variable"}
        )

        st.dataframe(
            numerical_summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No numerical variables are available."
        )

    st.write("---")


    st.subheader("Categorical Variable Summary")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) > 0:

        categorical_summary = (
            df[categorical_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        categorical_summary = categorical_summary.rename(
            columns={"index": "Variable"}
        )

        st.dataframe(
            categorical_summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No categorical variables are available."
        )

elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")

    st.write("""
    This page presents univariate, bivariate and multivariate analyses of the
    survey dataset. Use the filters to explore how demographic characteristics,
    online banking usage and cybersecurity-related scores vary across respondents.
    """)

    st.subheader("Interactive Filters")

    col1, col2, col3 = st.columns(3)

    with col1:

        age_options = df["Age"].dropna().unique().tolist()

        age_filter = st.multiselect(
            "Age Group",
            options=age_options,
            default=age_options
        )

    with col2:

        gender_options = df["Gender"].dropna().unique().tolist()

        gender_filter = st.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )

    with col3:

        banking_options = (
            df["Banking_Use"]
            .dropna()
            .unique()
            .tolist()
        )

        banking_filter = st.multiselect(
            "Banking Usage",
            options=banking_options,
            default=banking_options
        )

    filtered_df = df[
        df["Age"].isin(age_filter)
        & df["Gender"].isin(gender_filter)
        & df["Banking_Use"].isin(banking_filter)
    ].copy()

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters. "
            "Please change the selected options."
        )

    else:

        st.write("---")
        st.subheader("Filtered Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Responses",
            len(filtered_df)
        )

        c2.metric(
            "Average Awareness Score",
            f"{filtered_df['Awareness_Score'].mean():.2f}"
        )

        c3.metric(
            "Average Behaviour Score",
            f"{filtered_df['Behaviour_Score'].mean():.2f}"
        )

        c4.metric(
            "Average Biometric Score",
            f"{filtered_df['Biometric_Score'].mean():.2f}"
        )


        st.write("---")
        st.header("Univariate Analysis")

        st.write("""
        Univariate analysis examines one variable at a time. It is used to
        understand the frequency, distribution, central tendency and spread
        of individual variables.
        """)


        st.subheader("Categorical Variable Distributions")

        categorical_variable = st.selectbox(
            "Select a categorical variable",
            [
                "Age",
                "Gender",
                "Education",
                "Banking_Use",
                "Platform",
                "Awareness_Level",
                "Behaviour_Level",
                "Biometric_Level",
                "Biggest_Concern"
            ]
        )

        category_counts = (
            filtered_df[categorical_variable]
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.countplot(
                data=filtered_df,
                x=categorical_variable,
                order=category_counts.index,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {categorical_variable.replace('_', ' ')}"
            )

            ax.set_xlabel(
                categorical_variable.replace("_", " ")
            )

            ax.set_ylabel("Number of Respondents")

            plt.xticks(
                rotation=35,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.pie(
                category_counts.values,
                labels=category_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title(
                f"Percentage Distribution of "
                f"{categorical_variable.replace('_', ' ')}"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        st.subheader("Score Distribution")

        score_columns = [
            "Awareness_Score",
            "Opinion_Score",
            "Behaviour_Score",
            "Biometric_Score"
        ]

        score_variable = st.selectbox(
            "Select a score variable",
            score_columns,
            key="score_histogram_variable"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.histplot(
                filtered_df[score_variable].dropna(),
                bins=8,
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Histogram of {score_variable.replace('_', ' ')}"
            )

            ax.set_xlabel(
                score_variable.replace("_", " ")
            )

            ax.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.boxplot(
                y=filtered_df[score_variable],
                ax=ax
            )

            ax.set_title(
                f"Box Plot of {score_variable.replace('_', ' ')}"
            )

            ax.set_ylabel(
                score_variable.replace("_", " ")
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        st.subheader("Descriptive Statistics")

        descriptive_table = (
            filtered_df[score_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        descriptive_table = descriptive_table.rename(
            columns={"index": "Score Variable"}
        )

        st.dataframe(
            descriptive_table,
            use_container_width=True,
            hide_index=True
        )


        st.write("---")
        st.header("Bivariate Analysis")

        st.write("""Bivariate analysis examines the relationship between two variables.
        The charts below compare cybersecurity scores with demographic and
        online banking characteristics.
        """)

        st.subheader("Group Comparison")

        col1, col2 = st.columns(2)

        with col1:

            grouping_variable = st.selectbox(
                "Select a grouping variable",
                [
                    "Age",
                    "Gender",
                    "Education",
                    "Banking_Use",
                    "Platform"
                ]
            )

        with col2:

            comparison_score = st.selectbox(
                "Select a score to compare",
                score_columns,
                key="group_score"
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=filtered_df,
            x=grouping_variable,
            y=comparison_score,
            ax=ax
        )

        ax.set_title(
            f"{comparison_score.replace('_', ' ')} by "
            f"{grouping_variable.replace('_', ' ')}"
        )

        ax.set_xlabel(
            grouping_variable.replace("_", " ")
        )

        ax.set_ylabel(
            comparison_score.replace("_", " ")
        )

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Relationship Between Scores")

        col1, col2 = st.columns(2)

        with col1:

            x_score = st.selectbox(
                "Select X-axis score",
                score_columns,
                index=0
            )

        with col2:

            y_score = st.selectbox(
                "Select Y-axis score",
                score_columns,
                index=2
            )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.regplot(
            data=filtered_df,
            x=x_score,
            y=y_score,
            ax=ax
        )

        ax.set_title(
            f"Relationship Between "
            f"{x_score.replace('_', ' ')} and "
            f"{y_score.replace('_', ' ')}"
        )

        ax.set_xlabel(
            x_score.replace("_", " ")
        )

        ax.set_ylabel(
            y_score.replace("_", " ")
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        correlation_value = (
            filtered_df[[x_score, y_score]]
            .corr()
            .iloc[0, 1]
        )

        st.metric(
            "Pearson Correlation",
            f"{correlation_value:.3f}"
        )

        st.subheader("Awareness Level by Gender")

        stacked_table = pd.crosstab(
            filtered_df["Gender"],
            filtered_df["Awareness_Level"]
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        stacked_table.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_title(
            "Awareness Level by Gender"
        )

        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Respondents")

        plt.xticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write("---")
        st.header("Multivariate Analysis")

        st.write("""
        Multivariate analysis examines relationships among three or more
        variables. It helps identify broader patterns across awareness,
        behaviour, opinions and biometric understanding.
        """)

        st.subheader("Correlation Heatmap")

        correlation_matrix = (
            filtered_df[score_columns]
            .corr()
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Between Composite Scores"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Pairwise Score Analysis")

        pairplot_data = (
            filtered_df[score_columns]
            .dropna()
        )

        if len(pairplot_data) > 1:

            pair_grid = sns.pairplot(
                pairplot_data,
                diag_kind="hist"
            )

            pair_grid.fig.suptitle(
                "Pairwise Relationships Between Composite Scores",
                y=1.02
            )

            st.pyplot(pair_grid.fig)

            plt.close(pair_grid.fig)

        else:

            st.info(
                "Not enough records are available for pairwise analysis."
            )
        st.subheader("Bubble Chart")

        fig, ax = plt.subplots(figsize=(9, 6))

        bubble_sizes = (
            filtered_df["Opinion_Score"]
            .fillna(0)
            .clip(lower=0)
            * 100
        )

        scatter = ax.scatter(
            filtered_df["Awareness_Score"],
            filtered_df["Behaviour_Score"],
            s=bubble_sizes,
            alpha=0.6
        )

        ax.set_title(
            "Awareness, Behaviour and Opinion Scores"
        )

        ax.set_xlabel("Awareness Score")
        ax.set_ylabel("Behaviour Score")

        st.caption(
            "The position represents awareness and behaviour scores. "
            "Bubble size represents the opinion score."
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Average Composite Scores")

        average_scores = (
            filtered_df[score_columns]
            .mean()
            .reset_index()
        )

        average_scores.columns = [
            "Score Type",
            "Average Score"
        ]

        average_scores["Score Type"] = (
            average_scores["Score Type"]
            .str.replace("_Score", "", regex=False)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=average_scores,
            x="Score Type",
            y="Average Score",
            ax=ax
        )

        ax.set_title(
            "Average Composite Score Comparison"
        )

        ax.set_xlabel("Composite Measure")
        ax.set_ylabel("Average Score")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write("---")
        st.header("Key Findings")

        awareness_mean = (
            filtered_df["Awareness_Score"]
            .mean()
        )

        behaviour_mean = (
            filtered_df["Behaviour_Score"]
            .mean()
        )

        biometric_mean = (
            filtered_df["Biometric_Score"]
            .mean()
        )

        awareness_biometric_corr = (
            filtered_df[
                [
                    "Awareness_Score",
                    "Biometric_Score"
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        st.write(
            f"""The selected dataset contains **{len(filtered_df)} respondents**.
            The average awareness score is **{awareness_mean:.2f}**, while the
            average behaviour score is **{behaviour_mean:.2f}** and the average
            biometric score is **{biometric_mean:.2f}**.

            The correlation between awareness and biometric understanding is
            **{awareness_biometric_corr:.3f}**. The visualisations show how
            cybersecurity awareness and behaviour vary across demographic groups
            and online banking usage patterns.
            """
        )

        st.write("---")
        st.subheader("Filtered Dataset")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        filtered_csv = (
            filtered_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="Download Filtered Dataset",
            data=filtered_csv,
            file_name="Filtered_Data.csv",
            mime="text/csv"
        )

elif page == "Cyber Awareness Assessment":

    st.title("Cyber Awareness Assessment")

    st.write("""
    Complete the following assessment to evaluate your online banking
    cybersecurity awareness. Your answers will be used to calculate an
    awareness percentage and provide personalised recommendations.
    """)

    st.info(
        "This assessment is educational only and does not collect or store "
        "banking credentials or personal financial information."
    )

    score = 0
    total_score = 0
    feedback = []

    st.write("---")
    st.subheader("Section 1: Cybersecurity Knowledge")

    phishing_awareness = st.radio(
        "1. Can you recognise phishing emails pretending to be from a bank?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if phishing_awareness is not None:
        total_score += 2

        if phishing_awareness == "Strongly Agree":
            score += 2
        elif phishing_awareness == "Agree":
            score += 1.5
        elif phishing_awareness == "Neutral":
            score += 1
            feedback.append(
                "Learn common phishing warning signs such as suspicious links, "
                "urgent language and unusual sender addresses."
            )
        else:
            feedback.append(
                "Improve your ability to recognise phishing emails before "
                "responding or clicking links."
            )

    otp_awareness = st.radio(
        "2. Do you know that legitimate banks never ask customers to share OTPs, passwords or PINs?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if otp_awareness is not None:
        total_score += 2

        if otp_awareness == "Strongly Agree":
            score += 2
        elif otp_awareness == "Agree":
            score += 1.5
        elif otp_awareness == "Neutral":
            score += 1
            feedback.append(
                "Remember that OTPs, passwords and PINs must never be shared."
            )
        else:
            feedback.append(
                "Banks will not ask you to reveal OTPs, passwords or PINs."
            )

    mfa_awareness = st.radio(
        "3. Do you understand how Multi-Factor Authentication protects an online banking account?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if mfa_awareness is not None:
        total_score += 2

        if mfa_awareness == "Strongly Agree":
            score += 2
        elif mfa_awareness == "Agree":
            score += 1.5
        elif mfa_awareness == "Neutral":
            score += 1
            feedback.append(
                "Learn how MFA adds another verification step beyond a password."
            )
        else:
            feedback.append(
                "Improve your understanding of Multi-Factor Authentication."
            )

    website_awareness = st.radio(
        "4. Do you know how to verify that a banking website is genuine?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if website_awareness is not None:
        total_score += 2

        if website_awareness == "Strongly Agree":
            score += 2
        elif website_awareness == "Agree":
            score += 1.5
        elif website_awareness == "Neutral":
            score += 1
            feedback.append(
                "Check the website address carefully and access banking services "
                "through the official website or application."
            )
        else:
            feedback.append(
                "Learn how to verify a banking website before entering login details."
            )

    public_wifi_awareness = st.radio(
        "5. Do you understand the risks of using public Wi-Fi for online banking?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if public_wifi_awareness is not None:
        total_score += 2

        if public_wifi_awareness == "Strongly Agree":
            score += 2
        elif public_wifi_awareness == "Agree":
            score += 1.5
        elif public_wifi_awareness == "Neutral":
            score += 1
            feedback.append(
                "Avoid accessing online banking through unsecured public Wi-Fi."
            )
        else:
            feedback.append(
                "Public Wi-Fi may expose sensitive banking information."
            )

    ai_awareness = st.radio(
        "6. Are you aware of AI-generated scams and deepfake fraud targeting banking customers?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if ai_awareness is not None:
        total_score += 2

        if ai_awareness == "Strongly Agree":
            score += 2
        elif ai_awareness == "Agree":
            score += 1.5
        elif ai_awareness == "Neutral":
            score += 1
            feedback.append(
                "Learn how criminals may use artificial voices, videos and "
                "messages to impersonate trusted people."
            )
        else:
            feedback.append(
                "Improve your awareness of AI-generated scams and deepfake fraud."
            )

    compromised_account = st.radio(
        "7. Do you know what actions to take if your online banking account may have been compromised?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if compromised_account is not None:
        total_score += 2

        if compromised_account == "Strongly Agree":
            score += 2
        elif compromised_account == "Agree":
            score += 1.5
        elif compromised_account == "Neutral":
            score += 1
            feedback.append(
                "If you suspect fraud, change your password and contact your bank immediately."
            )
        else:
            feedback.append(
                "Learn the correct response to a potentially compromised account."
            )

    st.write("---")
    st.subheader("Practical Security Scenarios")

    phishing_scenario = st.radio(
        """8. You receive an email stating that your bank account will be
        suspended unless you click a link immediately. What would you do?
        """,
        [
            "Click the link immediately",
            "Ignore it completely",
            "Verify the request using the bank's official website or customer service",
            "Reply to the email"
        ],
        index=None
    )

    if phishing_scenario is not None:
        total_score += 2

        if phishing_scenario == (
            "Verify the request using the bank's official website or customer service"
        ):
            score += 2
        elif phishing_scenario == "Ignore it completely":
            score += 1
            feedback.append(
                "Ignoring the email avoids immediate danger, but you should also "
                "verify and report suspicious messages."
            )
        else:
            feedback.append(
                "Never click or reply to an urgent banking email without verification."
            )

    otp_scenario = st.radio(
        """9. Someone claiming to be from your bank asks for your OTP.
        What would you do?
        """,
        [
            "Share the OTP",
            "Ask why it is needed",
            "End the call and contact the bank through official channels",
            "Ignore the request but continue the conversation"
        ],
        index=None
    )

    if otp_scenario is not None:
        total_score += 2

        if otp_scenario == (
            "End the call and contact the bank through official channels"
        ):
            score += 2
        elif otp_scenario == "Ask why it is needed":
            score += 0.5
            feedback.append(
                "Do not continue discussing an OTP request. End the interaction "
                "and contact the bank independently."
            )
        else:
            feedback.append(
                "Never share an OTP or continue a suspicious banking conversation."
            )

    login_alert_scenario = st.radio(
        """10. You receive a security alert saying that someone logged into
        your account from another device. What is your first action?
        """,
        [
            "Ignore it",
            "Change your password immediately and contact the bank",
            "Wait to see if it happens again",
            "Delete the notification"
        ],
        index=None
    )

    if login_alert_scenario is not None:
        total_score += 2

        if login_alert_scenario == (
            "Change your password immediately and contact the bank"
        ):
            score += 2
        else:
            feedback.append(
                "Act immediately on an unauthorised login alert by securing the "
                "account and contacting the bank."
            )

    deepfake_scenario = st.radio(
        """11. You receive a voice call that sounds like your bank manager
        asking you to urgently approve a transaction. What would you do?
        """,
        [
            "Approve the transaction",
            "Verify the request using the bank's official contact details",
            "Share your banking details",
            "End the call without checking"
        ],
        index=None
    )

    if deepfake_scenario is not None:
        total_score += 2

        if deepfake_scenario == (
            "Verify the request using the bank's official contact details"
        ):
            score += 2
        elif deepfake_scenario == "End the call without checking":
            score += 1
            feedback.append(
                "Ending the call is safer, but the request should also be verified "
                "through official bank contact details."
            )
        else:
            feedback.append(
                "Voice calls can be imitated using AI. Verify every urgent request independently."
            )

    st.write("---")
    st.subheader("Online Banking Security Behaviour")

    unique_password = st.radio(
        "12. How often do you use a unique password for online banking?",
        ["Never", "Rarely", "Sometimes", "Always"],
        index=None
    )

    if unique_password is not None:
        total_score += 2

        if unique_password == "Always":
            score += 2
        elif unique_password == "Sometimes":
            score += 1
            feedback.append(
                "Use a unique password for online banking every time."
            )
        else:
            feedback.append(
                "Avoid reusing your online banking password on other accounts."
            )

    use_mfa = st.radio(
        "13. How often do you enable Multi-Factor Authentication when available?",
        ["Never", "Rarely", "Sometimes", "Always"],
        index=None
    )

    if use_mfa is not None:
        total_score += 2

        if use_mfa == "Always":
            score += 2
        elif use_mfa == "Sometimes":
            score += 1
            feedback.append(
                "Enable MFA whenever your bank provides the option."
            )
        else:
            feedback.append(
                "Multi-Factor Authentication adds important protection to your account."
            )

    update_application = st.radio(
        "14. How often do you update your banking application?",
        ["Never", "Rarely", "Sometimes", "Always"],
        index=None
    )

    if update_application is not None:
        total_score += 2

        if update_application == "Always":
            score += 2
        elif update_application == "Sometimes":
            score += 1
            feedback.append(
                "Install banking application updates promptly."
            )
        else:
            feedback.append(
                "Outdated banking applications may contain unresolved security weaknesses."
            )

    verify_website = st.radio(
        "15. How often do you verify website authenticity before entering banking credentials?",
        ["Never", "Rarely", "Sometimes", "Always"],
        index=None
    )

    if verify_website is not None:
        total_score += 2

        if verify_website == "Always":
            score += 2
        elif verify_website == "Sometimes":
            score += 1
            feedback.append(
                "Verify the banking website every time before entering credentials."
            )
        else:
            feedback.append(
                "Always confirm that you are using your bank's official website."
            )

    st.write("---")
    st.subheader("Biometric Authentication Awareness")

    biometric_understanding = st.radio(
        """16. Do you understand what biometric data, such as facial or
        fingerprint data, is used when logging into online banking?
        """,
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if biometric_understanding is not None:
        total_score += 2

        if biometric_understanding == "Strongly Agree":
            score += 2
        elif biometric_understanding == "Agree":
            score += 1.5
        elif biometric_understanding == "Neutral":
            score += 1
            feedback.append(
                "Learn how biometric features are converted into digital templates."
            )
        else:
            feedback.append(
                "Improve your understanding of how biometric authentication works."
            )

    biometric_protection = st.radio(
        "17. Do you understand how your biometric data should be stored and protected?",
        [
            "Strongly Agree",
            "Agree",
            "Neutral",
            "Disagree",
            "Strongly Disagree"
        ],
        index=None
    )

    if biometric_protection is not None:
        total_score += 2

        if biometric_protection == "Strongly Agree":
            score += 2
        elif biometric_protection == "Agree":
            score += 1.5
        elif biometric_protection == "Neutral":
            score += 1
            feedback.append(
                "Review how banks protect biometric templates and sensitive information."
            )
        else:
            feedback.append(
                "Learn more about biometric privacy, storage and data protection."
            )

    st.write("---")

    answered_questions = int(total_score / 2)
    required_questions = 17

    st.caption(
        f"Questions answered: {answered_questions} of {required_questions}"
    )

    if st.button("Calculate Awareness Score", type="primary"):

        if answered_questions < required_questions:
            st.error(
                "Please answer all 17 questions before calculating your score."
            )

        else:
            percentage = round((score / total_score) * 100, 1)

            st.write("---")
            st.subheader("Your Assessment Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Cyber Awareness Score",
                    f"{percentage}%"
                )

            with result_col2:
                if percentage >= 80:
                    awareness_level = "High"
                elif percentage >= 60:
                    awareness_level = "Moderate"
                else:
                    awareness_level = "Low"

                st.metric(
                    "Awareness Level",
                    awareness_level
                )

            st.progress(int(percentage))

            if percentage >= 80:
                st.success(
                    "You demonstrated a high level of online banking cybersecurity awareness."
                )

            elif percentage >= 60:
                st.warning(
                    "You demonstrated a moderate level of awareness, but some areas require improvement."
                )

            else:
                st.error(
                    "Your results indicate that further cybersecurity awareness training is recommended."
                )

            st.subheader("Personalised Recommendations")

            if feedback:
                unique_feedback = list(dict.fromkeys(feedback))

                for number, recommendation in enumerate(
                    unique_feedback,
                    start=1
                ):
                    st.write(f"{number}. {recommendation}")

            else:
                st.success(
                    "You demonstrated strong knowledge and safe behaviour across all assessed areas."
                )

            st.info(
                "Use the Learning Resources and Biometric Training Demo pages "
                "to strengthen any areas identified in your recommendations."
            )

elif page == "Biometric Training Demo":

    st.title("Biometric Facial Identification Training Demonstration")

    st.write("""
    This educational demonstration automatically checks whether a face is
    present in an uploaded photograph and displays facial landmarks around
    important facial features.
    """)

    st.warning("""
    This demonstration performs face and landmark detection only. It does not
    identify the person, store biometric data or perform banking authentication.
    """)

    biometric_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <style>
            * {
                box-sizing: border-box;
            }

            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                color: #f7f7f7;
                font-family: Arial, sans-serif;
            }

            body {
                padding: 6px;
            }

            h1, h2, h3, h4, p, ol, li, label, span {
                color: #f7f7f7;
            }

            p, li {
                font-size: 16px;
                line-height: 1.6;
            }

            .app-card {
                width: 100%;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 14px;
                padding: 22px;
            }

            .upload-box {
                background: #1f2630;
                border: 2px dashed #6b7280;
                border-radius: 12px;
                padding: 26px;
                text-align: center;
                margin-bottom: 18px;
            }

            .upload-box h3 {
                margin-top: 0;
                margin-bottom: 8px;
                color: #ffffff;
            }

            .upload-box p {
                margin-bottom: 16px;
                color: #d1d5db;
            }

            input[type="file"] {
                width: 100%;
                max-width: 440px;
                padding: 10px;
                color: #ffffff;
                background: #111827;
                border: 1px solid #4b5563;
                border-radius: 8px;
                cursor: pointer;
            }

            input[type="file"]::file-selector-button {
                margin-right: 12px;
                padding: 9px 14px;
                border: 0;
                border-radius: 7px;
                background: #ff4b4b;
                color: #ffffff;
                font-weight: 600;
                cursor: pointer;
            }

            #status {
                margin: 18px 0;
                padding: 14px 16px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 650;
            }

            .loading {
                background: #17365d;
                border: 1px solid #2f6fae;
                color: #ffffff;
            }

            .success {
                background: #123f2c;
                border: 1px solid #2c8a5b;
                color: #ffffff;
            }

            .error {
                background: #5a1f27;
                border: 1px solid #b84a58;
                color: #ffffff;
            }

            .image-wrapper {
                width: 100%;
                display: flex;
                justify-content: center;
                margin-top: 12px;
            }

            #uploadedImage,
            #outputCanvas {
                display: none;
                width: auto;
                max-width: 100%;
                max-height: 620px;
                border-radius: 12px;
                border: 1px solid #374151;
                background: #000000;
            }

            .results {
                display: none;
                margin-top: 26px;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 14px;
                margin: 16px 0 28px;
            }

            .metric {
                background: #1f2630;
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 18px 14px;
                text-align: center;
            }

            .metric-value {
                color: #58d6ff;
                font-size: 34px;
                font-weight: 750;
                line-height: 1.2;
            }

            .metric-label {
                margin-top: 7px;
                color: #c7cbd1;
                font-size: 14px;
            }

            .explanation {
                background: #1f2630;
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 18px 22px;
            }

            .explanation li,
            .explanation p {
                color: #e5e7eb;
            }

            .privacy-note {
                margin-top: 18px;
                padding: 14px 16px;
                border-radius: 10px;
                background: #24202f;
                border: 1px solid #5b4b73;
                color: #eee9f5;
            }

            @media (max-width: 700px) {
                .metric-grid {
                    grid-template-columns: 1fr;
                }

                .app-card {
                    padding: 14px;
                }
            }
        </style>
    </head>

    <body>
        <div class="app-card">

            <div class="upload-box">
                <h3>Upload a Photograph</h3>
                <p>Select a clear JPG or PNG image. The face should be visible and reasonably front-facing.</p>

                <input
                    type="file"
                    id="imageUpload"
                    accept="image/jpeg,image/png"
                >
            </div>

            <div id="status" class="loading">
                Loading facial landmark detection model...
            </div>

            <div class="image-wrapper">
                <img id="uploadedImage" alt="Uploaded photograph">
                <canvas id="outputCanvas"></canvas>
            </div>

            <div id="results" class="results">

                <h3>Detection Results</h3>

                <div class="metric-grid">
                    <div class="metric">
                        <div id="faceCount" class="metric-value">0</div>
                        <div class="metric-label">Faces Detected</div>
                    </div>

                    <div class="metric">
                        <div id="landmarkCount" class="metric-value">0</div>
                        <div class="metric-label">Landmarks Detected</div>
                    </div>

                    <div class="metric">
                        <div id="detectionResult" class="metric-value">No</div>
                        <div class="metric-label">Face Present</div>
                    </div>
                </div>

                <div class="explanation">
                    <h3>How Face Detection Works</h3>

                    <ol>
                        <li>The selected photograph is loaded in the user's browser.</li>
                        <li>The facial landmark model analyses the image.</li>
                        <li>The model checks whether one or more faces are present.</li>
                        <li>Facial landmark coordinates are generated.</li>
                        <li>The detected landmarks are drawn over the photograph.</li>
                    </ol>

                    <p>
                        In a complete biometric authentication system, facial
                        characteristics may be processed further to generate a
                        protected biometric template or face embedding.
                    </p>
                </div>

                <div class="privacy-note">
                    The selected photograph is processed for this visual demonstration.
                    This page does not create a face database or identify the person.
                </div>

            </div>
        </div>

        <script type="module">

            import {
                FaceLandmarker,
                FilesetResolver,
                DrawingUtils
            } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

            const statusBox = document.getElementById("status");
            const uploadInput = document.getElementById("imageUpload");
            const imageElement = document.getElementById("uploadedImage");
            const canvas = document.getElementById("outputCanvas");
            const canvasContext = canvas.getContext("2d");
            const resultsSection = document.getElementById("results");
            const faceCount = document.getElementById("faceCount");
            const landmarkCount = document.getElementById("landmarkCount");
            const detectionResult = document.getElementById("detectionResult");

            let faceLandmarker = null;

            function setStatus(type, message) {
                statusBox.className = type;
                statusBox.textContent = message;
            }

            async function initialiseFaceLandmarker() {
                try {
                    const vision = await FilesetResolver.forVisionTasks(
                        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                    );

                    faceLandmarker = await FaceLandmarker.createFromOptions(
                        vision,
                        {
                            baseOptions: {
                                modelAssetPath:
                                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                                delegate: "GPU"
                            },
                            runningMode: "IMAGE",
                            numFaces: 5,
                            minFaceDetectionConfidence: 0.5,
                            minFacePresenceConfidence: 0.5,
                            minTrackingConfidence: 0.5
                        }
                    );

                    setStatus(
                        "success",
                        "The face detection model is ready. Upload a photograph."
                    );

                } catch (error) {
                    console.error(error);

                    setStatus(
                        "error",
                        "The facial detection model could not be loaded. Refresh the page and try again."
                    );
                }
            }

            uploadInput.addEventListener("change", function(event) {
                const selectedFile = event.target.files[0];

                if (!selectedFile) {
                    return;
                }

                if (!faceLandmarker) {
                    setStatus(
                        "error",
                        "The model is still loading. Wait a moment and select the image again."
                    );
                    return;
                }

                const fileReader = new FileReader();

                fileReader.onload = function(loadEvent) {
                    imageElement.onload = function() {
                        detectFace();
                    };

                    imageElement.src = loadEvent.target.result;
                };

                fileReader.readAsDataURL(selectedFile);
            });

            function detectFace() {
                setStatus("loading", "Analysing the photograph...");

                try {
                    const detection = faceLandmarker.detect(imageElement);

                    canvas.width = imageElement.naturalWidth;
                    canvas.height = imageElement.naturalHeight;

                    canvasContext.clearRect(
                        0,
                        0,
                        canvas.width,
                        canvas.height
                    );

                    canvasContext.drawImage(
                        imageElement,
                        0,
                        0,
                        canvas.width,
                        canvas.height
                    );

                    const detectedLandmarks = detection.faceLandmarks || [];

                    resultsSection.style.display = "block";

                    if (detectedLandmarks.length === 0) {
                        canvas.style.display = "none";
                        imageElement.style.display = "block";

                        faceCount.textContent = "0";
                        landmarkCount.textContent = "0";
                        detectionResult.textContent = "No";

                        setStatus(
                            "error",
                            "No face was detected. Upload a clearer photograph with a visible face."
                        );

                        return;
                    }

                    const drawingUtils = new DrawingUtils(canvasContext);

                    for (const landmarks of detectedLandmarks) {
                        drawingUtils.drawConnectors(
                            landmarks,
                            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                            {color: "#b9c0c766", lineWidth: 1}
                        );

                        drawingUtils.drawConnectors(
                            landmarks,
                            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
                            {color: "#ff5353", lineWidth: 2}
                        );

                        drawingUtils.drawConnectors(
                            landmarks,
                            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
                            {color: "#42e36f", lineWidth: 2}
                        );

                        drawingUtils.drawConnectors(
                            landmarks,
                            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
                            {color: "#f0f2f5", lineWidth: 2}
                        );

                        drawingUtils.drawConnectors(
                            landmarks,
                            FaceLandmarker.FACE_LANDMARKS_LIPS,
                            {color: "#f05be8", lineWidth: 2}
                        );
                    }

                    imageElement.style.display = "none";
                    canvas.style.display = "block";

                    faceCount.textContent = detectedLandmarks.length;

                    landmarkCount.textContent = detectedLandmarks.reduce(
                        function(total, landmarks) {
                            return total + landmarks.length;
                        },
                        0
                    );

                    detectionResult.textContent = "Yes";

                    setStatus(
                        "success",
                        detectedLandmarks.length +
                        " face(s) detected successfully."
                    );

                } catch (error) {
                    console.error(error);

                    canvas.style.display = "none";
                    imageElement.style.display = "block";

                    setStatus(
                        "error",
                        "The image could not be analysed. Try another clear JPG or PNG photograph."
                    );
                }
            }

            initialiseFaceLandmarker();

        </script>
    </body>
    </html>
    """

    components.html(
        biometric_html,
        height=1200,
        scrolling=True
    )

    st.write("---")


    st.subheader("Privacy Considerations")

    st.write("""
    The photograph is selected and processed in the browser for the visual
    demonstration. The application does not add it to the research dataset,
    associate it with a banking account or create a permanent biometric record.

    This page intentionally stops at landmark detection because no biometric
    database is used in this research prototype.
    """)


elif page == "Learning Resources":

    st.title("Personalised Learning Resources")

    st.write("""
    Select the score you received from the Cyber Awareness Assessment.
    The toolkit will recommend learning modules based on your awareness level.
    """)

    score_range = st.selectbox(
        "Select your Cyber Awareness Assessment Score",
        [
            "Select score",
            "0–20%",
            "21–40%",
            "41–60%",
            "61–80%",
            "81–100%"
        ]
    )

    if score_range == "Select score":

        st.info(
            "Complete the Cyber Awareness Assessment first, then select "
            "your score range to receive personalised learning."
        )

    else:

        if score_range == "0–20%":

            awareness_level = "Very Low Awareness"

            message = """Your result indicates that you currently have limited knowledge
            of online banking cybersecurity. Complete all recommended modules
            in the suggested order.
            """

            modules = [
                "Cybersecurity Basics",
                "Phishing and Scam Detection",
                "Password and MFA Security",
                "Safe Online Banking Practices",
                "Biometric Authentication",
                "AI and Deepfake Fraud"
            ]

        elif score_range == "21–40%":

            awareness_level = "Low Awareness"

            message = """You understand some basic cybersecurity concepts, but several
            important areas still require improvement.
            """

            modules = [
                "Phishing and Scam Detection",
                "Password and MFA Security",
                "Safe Online Banking Practices",
                "Biometric Authentication",
                "AI and Deepfake Fraud"
            ]

        elif score_range == "41–60%":

            awareness_level = "Moderate Awareness"

            message = """You demonstrate a reasonable understanding of online banking
            security. The following modules will strengthen your knowledge.
            """

            modules = [
                "Advanced Phishing Detection",
                "Safe Online Banking Practices",
                "Biometric Authentication",
                "AI and Deepfake Fraud",
                "Emerging Cyber Threats"
            ]

        elif score_range == "61–80%":

            awareness_level = "Good Awareness"

            message = """You already follow many secure online banking practices.
            Focus on advanced and emerging cybersecurity threats.
            """

            modules = [
                "Biometric Authentication",
                "AI and Deepfake Fraud",
                "Emerging Cyber Threats",
                "Advanced Security Practices"
            ]

        else:

            awareness_level = "High Awareness"

            message = """You demonstrate a high level of cybersecurity awareness.
            Continue reviewing advanced security practices and emerging threats.
            """

            modules = [
                "Emerging AI Threats",
                "Advanced Security Practices",
                "Biometric Security and Privacy",
                "Current Online Banking Threats"
            ]

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Selected Score Range",
                score_range
            )

        with col2:
            st.metric(
                "Awareness Level",
                awareness_level
            )

        if score_range == "0–20%":
            st.error(message)

        elif score_range == "21–40%":
            st.warning(message)

        elif score_range == "41–60%":
            st.info(message)

        else:
            st.success(message)

        st.subheader("Recommended Learning Path")

        for number, module_name in enumerate(modules, start=1):
            st.write(f"{number}. {module_name}")

        st.write("---")

        selected_module = st.selectbox(
            "Select a Learning Module",
            modules
        )

        st.subheader(selected_module)

        # -------------------------------------------------
        # MODULE 1: CYBERSECURITY BASICS
        # -------------------------------------------------

        if selected_module == "Cybersecurity Basics":

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Explain what cybersecurity means.
            - Understand why online banking users are targeted.
            - Recognise common online banking threats.
            - Understand why personal information must be protected.
            """)

            st.markdown("""
            ### Learning Content

            Cybersecurity is the protection of computers, mobile devices,
            online accounts, networks and personal information from
            unauthorised access and cyberattacks.

            Online banking users are targeted because banking accounts contain
            valuable financial and personal information. Cybercriminals may use
            phishing emails, fake websites, malicious applications, social
            engineering and AI-generated scams to steal this information.

            Users can reduce risk by following secure banking practices,
            protecting login details and responding carefully to suspicious
            messages.
            """)

            question = st.radio(
                "What is the main purpose of cybersecurity?",
                [
                    "To increase internet speed",
                    "To protect systems and personal information",
                    "To make banking applications faster"
                ],
                index=None,
                key="basics_question"
            )

            if st.button(
                "Check Answer",
                key="check_basics"
            ):

                if question is None:
                    st.warning("Select an answer first.")

                elif question == (
                    "To protect systems and personal information"
                ):
                    st.success(
                        "Correct. Cybersecurity protects systems, accounts "
                        "and personal information."
                    )

                else:
                    st.error(
                        "Incorrect. The main purpose of cybersecurity is to "
                        "protect systems and personal information."
                    )

        
        elif selected_module in [
            "Phishing and Scam Detection",
            "Advanced Phishing Detection"
        ]:

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Recognise suspicious banking emails.
            - Identify fake banking websites.
            - Detect SMS and voice phishing attempts.
            - Verify suspicious requests safely.
            """)

            st.markdown("""
            ### Learning Content

            Phishing is a cyberattack in which criminals pretend to be a trusted
            organisation, such as a bank, to steal personal information.

            Warning signs may include urgent language, unfamiliar sender
            addresses, spelling errors, suspicious links and requests for
            passwords, PINs or OTPs.

            Never use a link in a suspicious message. Open your bank's official
            application or type the official website address directly into your
            browser.
            """)

            phishing_question = st.radio(
                """You receive an email saying your bank account will be suspended
                unless you click a link. What should you do?
                """,
                [
                    "Click the link immediately",
                    "Reply with your banking details",
                    "Verify the message using the bank's official website or contact details"
                ],
                index=None,
                key="phishing_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_phishing_learning"
            ):

                if phishing_question is None:
                    st.warning("Select an answer first.")

                elif phishing_question == (
                    "Verify the message using the bank's official website or contact details"
                ):
                    st.success(
                        "Correct. Suspicious banking messages should always "
                        "be verified independently."
                    )

                else:
                    st.error(
                        "Incorrect. Never click suspicious links or share banking details."
                    )

  
        elif selected_module == "Password and MFA Security":

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Create strong and unique passwords.
            - Avoid password reuse.
            - Understand Multi-Factor Authentication.
            - Protect OTPs and authentication codes.
            """)

            st.markdown("""
            ### Learning Content

            A strong password should be difficult to guess and should not be
            reused across different accounts. Password reuse increases the risk
            of credential stuffing attacks.

            Multi-Factor Authentication adds another verification step beyond a
            password. This may include an authenticator application, biometric
            verification or a security key.

            OTPs, PINs and passwords must never be shared with another person,
            including someone claiming to work for a bank.
            """)

            mfa_question = st.radio(
                """Someone claiming to be from your bank asks for your OTP.
                What should you do?
                """,
                [
                    "Share the OTP",
                    "Ask why they need it",
                    "End the interaction and contact the bank through official channels"
                ],
                index=None,
                key="mfa_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_mfa_learning"
            ):

                if mfa_question is None:
                    st.warning("Select an answer first.")

                elif mfa_question == (
                    "End the interaction and contact the bank through official channels"
                ):
                    st.success(
                        "Correct. OTPs must never be shared."
                    )

                else:
                    st.error(
                        "Incorrect. End the interaction and verify the request "
                        "through official bank channels."
                    )

      
        elif selected_module == "Safe Online Banking Practices":

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Use online banking safely.
            - Verify banking websites.
            - Avoid unsafe public Wi-Fi.
            - Respond correctly to security alerts.
            - Keep banking applications updated.
            """)

            st.markdown("""
            ### Learning Content

            Use your bank's official application or website and carefully check
            the website address before entering login details.

            Avoid using unsecured public Wi-Fi for online banking. Keep your
            banking application and device operating system updated.

            Security alerts should be taken seriously. If you receive an
            unfamiliar login notification, change your password immediately and
            contact your bank.
            """)

            safe_question = st.radio(
                """ You receive an alert showing an unfamiliar login to your account.
                What should you do first?
                """,
                [
                    "Ignore it",
                    "Wait to see whether it happens again",
                    "Change your password and contact the bank"
                ],
                index=None,
                key="safe_banking_question"
            )

            if st.button(
                "Check Answer",
                key="check_safe_banking"
            ):

                if safe_question is None:
                    st.warning("Select an answer first.")

                elif safe_question == (
                    "Change your password and contact the bank"
                ):
                    st.success(
                        "Correct. An unfamiliar login alert requires immediate action."
                    )

                else:
                    st.error(
                        "Incorrect. Secure the account immediately and contact the bank."
                    )

        elif selected_module in [
            "Biometric Authentication",
            "Biometric Security and Privacy"
        ]:

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Understand facial and fingerprint authentication.
            - Explain the purpose of facial landmarks.
            - Understand biometric templates.
            - Recognise biometric privacy risks.
            """)

            st.markdown("""
            ### Learning Content

            Biometric authentication verifies users using physical or behavioural
            characteristics such as fingerprints, facial features or voice
            patterns.

            In facial systems, a model first detects a face and identifies facial
            landmarks around features such as the eyes, nose, mouth and facial
            outline.

            In a complete authentication system, these characteristics may be
            converted into a mathematical biometric template. The template must
            be protected because biometric information cannot easily be replaced
            if compromised.

            Use the Biometric Training Demo page to view automatic face and
            landmark detection.
            """)

            biometric_question = st.radio(
                """What should a secure biometric system protect?
                """,
                [
                    "Only the screen colour",
                    "The biometric template and related personal data",
                    "The user's internet speed"
                ],
                index=None,
                key="biometric_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_biometric_learning"
            ):

                if biometric_question is None:
                    st.warning("Select an answer first.")

                elif biometric_question == (
                    "The biometric template and related personal data"
                ):
                    st.success(
                        "Correct. Biometric templates and personal data require strong protection."
                    )

                else:
                    st.error(
                        "Incorrect. Biometric templates and personal information "
                        "must be secured."
                    )

   
        elif selected_module in [
            "AI and Deepfake Fraud",
            "Emerging AI Threats"
        ]:

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Recognise AI-generated scams.
            - Understand deepfake voice and video risks.
            - Verify urgent financial requests.
            - Respond safely to suspicious impersonation.
            """)

            st.markdown("""
            ### Learning Content

            Artificial Intelligence can be used to create realistic fake voices,
            videos, emails and messages.

            A criminal may imitate a bank employee, manager, friend or family
            member and create urgency to persuade the victim to transfer money or
            reveal sensitive information.

            Do not rely only on a familiar voice or appearance. Verify urgent
            requests using official contact details or another trusted method.
            """)

            ai_question = st.radio(
                """A caller sounds exactly like your bank manager and asks you to
                approve an urgent transfer. What should you do?
                """,
                [
                    "Approve the transfer",
                    "Share your banking details",
                    "End the call and verify through the bank's official contact details"
                ],
                index=None,
                key="ai_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_ai_learning"
            ):

                if ai_question is None:
                    st.warning("Select an answer first.")

                elif ai_question == (
                    "End the call and verify through the bank's official contact details"
                ):
                    st.success(
                        "Correct. Voice and video can be imitated using AI."
                    )

                else:
                    st.error(
                        "Incorrect. Always verify urgent financial requests independently."
                    )

      
        elif selected_module in [
            "Emerging Cyber Threats",
            "Current Online Banking Threats"
        ]:

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to recognise:

            - Malware and banking trojans.
            - Credential stuffing.
            - SIM-swapping.
            - Man-in-the-Middle attacks.
            - AI-assisted phishing.
            """)

            st.markdown("""
            ### Learning Content

            Malware may steal login details or monitor banking sessions.
            Credential stuffing uses passwords exposed in previous data breaches.
            SIM-swapping allows criminals to intercept SMS authentication codes.

            Man-in-the-Middle attacks may intercept communication, particularly
            when users connect through unsafe networks.

            Protect yourself by using unique passwords, enabling strong MFA,
            updating devices, avoiding suspicious downloads and monitoring
            banking activity.
            """)

            threat_question = st.radio(
                """Which action reduces the risk of credential stuffing?
                """,
                [
                    "Reuse the same password everywhere",
                    "Use a unique password for online banking",
                    "Share passwords with trusted people"
                ],
                index=None,
                key="threat_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_threat_learning"
            ):

                if threat_question is None:
                    st.warning("Select an answer first.")

                elif threat_question == (
                    "Use a unique password for online banking"
                ):
                    st.success(
                        "Correct. Unique passwords reduce credential-stuffing risk."
                    )

                else:
                    st.error(
                        "Incorrect. Online banking should use a strong, unique password."
                    )

   
        elif selected_module == "Advanced Security Practices":

            st.markdown("""
            ### Learning Objectives

            By completing this module, you should be able to:

            - Use phishing-resistant authentication.
            - Review account activity.
            - Respond safely to authentication prompts.
            - Understand shared responsibility for banking security.
            """)

            st.markdown("""
            ### Learning Content

            Advanced security practices include using phishing-resistant
            authentication methods where available, reviewing banking activity,
            checking login alerts and avoiding approval of unexpected
            authentication requests.

            Banks provide security controls, but customers must use them correctly.
            Secure online banking therefore depends on both reliable technology
            and informed user behaviour.
            """)

            advanced_question = st.radio(
                """What should you do when you receive an unexpected MFA approval request?
                """,
                [
                    "Approve it to stop the notifications",
                    "Reject it and secure the account",
                    "Ignore all future MFA requests"
                ],
                index=None,
                key="advanced_learning_question"
            )

            if st.button(
                "Check Answer",
                key="check_advanced_learning"
            ):

                if advanced_question is None:
                    st.warning("Select an answer first.")

                elif advanced_question == (
                    "Reject it and secure the account"
                ):
                    st.success(
                        "Correct. Unexpected authentication requests may indicate an attack."
                    )

                else:
                    st.error(
                        "Incorrect. Reject unexpected requests and review account security."
                    )

        st.write("---")
        st.subheader("Expected Learning Outcome")

        if score_range == "0–20%":

            st.write("""After completing all recommended modules, the user should be able to
            recognise basic cyber threats, protect passwords and OTPs, respond to
            suspicious messages and adopt safer online banking behaviour.
            """)

        elif score_range == "21–40%":

            st.write("""
            After completing the recommended modules, the user should strengthen
            their ability to recognise scams, use MFA correctly and respond to
            suspicious account activity.
            """)

        elif score_range == "41–60%":

            st.write("""After completing the recommended modules, the user should develop a
            stronger understanding of biometric security, AI-enabled fraud and
            emerging online banking threats.
            """)

        elif score_range == "61–80%":

            st.write("""After completing the recommended modules, the user should improve
            their knowledge of advanced authentication, biometric privacy and
            emerging cyber threats.
            """)

        else:

            st.write("""After completing the recommended modules, the user should maintain
            high cybersecurity awareness and remain informed about evolving
            threats and advanced security practices.
            """)
