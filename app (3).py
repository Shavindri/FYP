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

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes)



elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")
    st.write(
        "Use the filters below to explore demographics, online banking usage, "
        "cybersecurity awareness, behaviour, and biometric understanding."
    )

    st.subheader("Interactive Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_options = df["Age"].dropna().unique().tolist()
        age_filter = st.multiselect(
            "Age",
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
        banking_options = df["Banking_Use"].dropna().unique().tolist()
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
        st.warning("No records match the selected filters. Please change the filters.")
    else:
        st.subheader("Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Responses", len(filtered_df))

        awareness_average = filtered_df["Awareness_Score"].mean()
        c2.metric(
            "Average Awareness",
            f"{awareness_average:.2f}" if pd.notna(awareness_average) else "N/A"
        )

        c3.metric(
            "Mobile App Users",
            int(filtered_df["Platform"].value_counts().get("Mobile App", 0))
        )

        c4.metric(
            "High Awareness",
            int(filtered_df["Awareness_Level"].value_counts().get("High", 0))
        )

        st.write("---")
        st.subheader("Demographic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Age Distribution")
            st.bar_chart(filtered_df["Age"].value_counts())

        with col2:
            st.markdown("#### Gender Distribution")
            st.bar_chart(filtered_df["Gender"].value_counts())

        st.write("---")
        st.subheader("Online Banking Usage")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Banking Usage Frequency")
            st.bar_chart(filtered_df["Banking_Use"].value_counts())

        with col2:
            st.markdown("#### Platform Usage")
            st.bar_chart(filtered_df["Platform"].value_counts())

        st.write("---")
        st.subheader("Cybersecurity Levels")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Awareness Level")
            st.bar_chart(filtered_df["Awareness_Level"].value_counts())

        with col2:
            st.markdown("#### Behaviour Level")
            st.bar_chart(filtered_df["Behaviour_Level"].value_counts())

        st.write("---")
        st.subheader("Biometric Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Biometric Understanding")
            st.bar_chart(filtered_df["Biometric_Level"].value_counts())

        with col2:
            st.markdown("#### Biggest Security Concern")
            st.bar_chart(filtered_df["Biggest_Concern"].value_counts())

        st.write("---")
        st.subheader("Correlation Between Scores")

        corr_cols = [
            "Awareness_Score",
            "Opinion_Score",
            "Behaviour_Score",
            "Biometric_Score"
        ]

        available_corr_cols = [
            column for column in corr_cols
            if column in filtered_df.columns
        ]

        if len(available_corr_cols) >= 2:
            corr = filtered_df[available_corr_cols].corr()

            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                corr,
                annot=True,
                cmap="Blues",
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar=False,
                annot_kws={"size": 9},
                ax=ax
            )

            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0, labelsize=8)
            plt.tight_layout()

            left, centre, right = st.columns([1, 2, 1])

            with centre:
                st.pyplot(fig)

            plt.close(fig)
        else:
            st.info("Not enough score columns are available for correlation analysis.")

        st.write("---")
        st.subheader("Filtered Dataset")

        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Dataset",
            data=csv,
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
    This educational demonstration automatically detects whether a face is
    present in an uploaded photograph and displays facial landmarks around
    important facial features.
    """)

    st.warning("""
    This demonstration performs face and landmark detection only. It does not
    identify the person, store biometric data or perform banking authentication.
    """)

    biometric_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">

        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 10px;
                background-color: transparent;
            }

            .upload-box {
                border: 2px dashed #999;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }

            input[type="file"] {
                margin: 10px;
            }

            #status {
                margin: 15px 0;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
            }

            .loading {
                background-color: #e8f1ff;
                color: #174ea6;
            }

            .success {
                background-color: #e7f7ed;
                color: #137333;
            }

            .error {
                background-color: #fce8e6;
                color: #c5221f;
            }

            .image-container {
                position: relative;
                display: inline-block;
                max-width: 100%;
            }

            #uploadedImage {
                display: none;
                max-width: 100%;
                max-height: 600px;
                border-radius: 8px;
            }

            #outputCanvas {
                display: none;
                max-width: 100%;
                max-height: 600px;
                border-radius: 8px;
            }

            .results {
                margin-top: 20px;
                display: none;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-top: 15px;
            }

            .metric {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }

            .metric-value {
                font-size: 24px;
                font-weight: bold;
            }

            .metric-label {
                font-size: 13px;
                color: #555;
            }

            @media (max-width: 700px) {
                .metric-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>

    <body>

        <div class="upload-box">
            <h3>Upload a Photograph</h3>

            <p>
                Select a clear JPG or PNG image containing a front-facing face.
            </p>

            <input
                type="file"
                id="imageUpload"
                accept="image/jpeg,image/png"
            >
        </div>

        <div id="status" class="loading">
            Loading facial landmark detection model...
        </div>

        <div class="image-container">
            <img id="uploadedImage">
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

            <h3>How Detection Works</h3>

            <ol>
                <li>The photograph is loaded into the browser.</li>
                <li>The facial landmark model examines the image.</li>
                <li>The model checks whether a face is present.</li>
                <li>Facial landmark coordinates are generated.</li>
                <li>The detected landmarks are drawn over the photograph.</li>
            </ol>

            <p>
                In a complete biometric authentication system, these facial
                characteristics may be processed further to generate a protected
                biometric template or face embedding.
            </p>

        </div>

        <script type="module">

            import {
                FaceLandmarker,
                FilesetResolver,
                DrawingUtils
            } from
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

            const statusBox =
                document.getElementById("status");

            const uploadInput =
                document.getElementById("imageUpload");

            const imageElement =
                document.getElementById("uploadedImage");

            const canvas =
                document.getElementById("outputCanvas");

            const canvasContext =
                canvas.getContext("2d");

            const resultsSection =
                document.getElementById("results");

            const faceCount =
                document.getElementById("faceCount");

            const landmarkCount =
                document.getElementById("landmarkCount");

            const detectionResult =
                document.getElementById("detectionResult");

            let faceLandmarker;

            async function initialiseFaceLandmarker() {

                try {

                    const vision =
                        await FilesetResolver.forVisionTasks(
                            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                        );

                    faceLandmarker =
                        await FaceLandmarker.createFromOptions(
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

                    statusBox.className = "success";

                    statusBox.textContent =
                        "The face detection model is ready. Upload a photograph.";

                } catch (error) {

                    console.error(error);

                    statusBox.className = "error";

                    statusBox.textContent =
                        "The facial detection model could not be loaded. Refresh the page and try again.";
                }
            }

            uploadInput.addEventListener(
                "change",
                function(event) {

                    const selectedFile =
                        event.target.files[0];

                    if (!selectedFile) {
                        return;
                    }

                    if (!faceLandmarker) {

                        statusBox.className = "error";

                        statusBox.textContent =
                            "The model is still loading. Wait a moment and try again.";

                        return;
                    }

                    const fileReader =
                        new FileReader();

                    fileReader.onload =
                        function(loadEvent) {

                            imageElement.onload =
                                function() {

                                    detectFace();
                                };

                            imageElement.src =
                                loadEvent.target.result;
                        };

                    fileReader.readAsDataURL(
                        selectedFile
                    );
                }
            );

            function detectFace() {

                statusBox.className = "loading";

                statusBox.textContent =
                    "Analysing the photograph...";

                const detection =
                    faceLandmarker.detect(
                        imageElement
                    );

                canvas.width =
                    imageElement.naturalWidth;

                canvas.height =
                    imageElement.naturalHeight;

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

                const detectedLandmarks =
                    detection.faceLandmarks || [];

                resultsSection.style.display =
                    "block";

                if (detectedLandmarks.length === 0) {

                    canvas.style.display =
                        "none";

                    imageElement.style.display =
                        "block";

                    statusBox.className =
                        "error";

                    statusBox.textContent =
                        "No face was detected in the uploaded photograph.";

                    faceCount.textContent =
                        "0";

                    landmarkCount.textContent =
                        "0";

                    detectionResult.textContent =
                        "No";

                    return;
                }

                const drawingUtils =
                    new DrawingUtils(
                        canvasContext
                    );

                for (
                    const landmarks of detectedLandmarks
                ) {

                    drawingUtils.drawConnectors(
                        landmarks,
                        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                        {
                            color: "#C0C0C070",
                            lineWidth: 1
                        }
                    );

                    drawingUtils.drawConnectors(
                        landmarks,
                        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
                        {
                            color: "#FF3030",
                            lineWidth: 2
                        }
                    );

                    drawingUtils.drawConnectors(
                        landmarks,
                        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
                        {
                            color: "#30FF30",
                            lineWidth: 2
                        }
                    );

                    drawingUtils.drawConnectors(
                        landmarks,
                        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
                        {
                            color: "#E0E0E0",
                            lineWidth: 2
                        }
                    );

                    drawingUtils.drawConnectors(
                        landmarks,
                        FaceLandmarker.FACE_LANDMARKS_LIPS,
                        {
                            color: "#E030E0",
                            lineWidth: 2
                        }
                    );
                }

                imageElement.style.display =
                    "none";

                canvas.style.display =
                    "block";

                statusBox.className =
                    "success";

                statusBox.textContent =
                    detectedLandmarks.length +
                    " face(s) detected successfully.";

                faceCount.textContent =
                    detectedLandmarks.length;

                landmarkCount.textContent =
                    detectedLandmarks.reduce(
                        function(total, landmarks) {
                            return total + landmarks.length;
                        },
                        0
                    );

                detectionResult.textContent =
                    "Yes";
            }

            initialiseFaceLandmarker();

        </script>

    </body>
    </html>
    """

    components.html(
        biometric_html,
        height=1100,
        scrolling=True
    )

    st.write("---")

    st.subheader("Academic Limitation")

    st.write("""
    This demonstration detects faces and facial landmarks but does not perform
    facial identification. Facial identification would require the detected
    features to be converted into a secure biometric embedding and compared
    against an enrolled reference template.

    This page intentionally stops at landmark detection because no biometric
    database is used in this research prototype.
    """)

    st.subheader("Privacy Considerations")

    st.write("""
    The browser processes the selected photograph for the visual demonstration.
    The application does not add the photograph to the research dataset and does
    not associate the face with a banking account.
    """)

elif page == "Learning Resources":
    st.title("Learning Resources")

    topic = st.selectbox(
        "Choose a topic",
        [
            "Phishing Awareness",
            "OTP Security",
            "Password Security",
            "Multi-Factor Authentication",
            "Safe Online Banking Behaviour",
            "Biometric Authentication",
            "AI Scams and Deepfake Fraud"
        ]
    )

    if topic == "Phishing Awareness":
        st.write("""
        Phishing is a method used by cybercriminals to trick users into revealing sensitive information.
        Users should avoid clicking suspicious links and should verify messages through official bank channels.
        """)

    elif topic == "OTP Security":
        st.write("""
        OTPs should never be shared with anyone. Legitimate banks do not ask customers to reveal OTPs,
        passwords, or PINs.
        """)

    elif topic == "Password Security":
        st.write("""
        Online banking users should use strong, unique passwords and avoid reusing passwords across accounts.
        """)

    elif topic == "Multi-Factor Authentication":
        st.write("""
        Multi-Factor Authentication strengthens account security by requiring more than one method of verification.
        """)

    elif topic == "Safe Online Banking Behaviour":
        st.write("""
        Safe online banking behaviour includes avoiding public Wi-Fi, updating banking apps,
        verifying website authenticity, and monitoring security alerts.
        """)

    elif topic == "Biometric Authentication":
        st.write("""
        Biometric authentication uses unique physical characteristics such as fingerprints or facial recognition.
        Users should understand how biometric data is stored, protected, and used.
        """)

    elif topic == "AI Scams and Deepfake Fraud":
        st.write("""
        AI-generated scams and deepfake fraud can imitate real voices, images, or messages.
        Users should verify urgent financial requests through official banking channels.
        """)
