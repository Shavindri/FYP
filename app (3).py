import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    This educational demonstration explains the basic facial identification
    process: an image is uploaded, illustrative facial landmarks are displayed,
    a simplified numerical faceprint is generated, and a second image is
    compared with the stored reference.
    """)

    st.warning("""
    This is a simplified educational simulation. It does not perform real facial
    recognition and must not be used for identity verification or banking access.
    """)

    # -----------------------------------------------------
    # FUNCTIONS USED ONLY ON THIS PAGE
    # -----------------------------------------------------

    def prepare_image(uploaded_file):
        """Open and resize an uploaded image."""

        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((700, 700))

        return image


    def draw_simulated_landmarks(image):
        """
        Draw illustrative landmark points using positions relative to
        the image size. These are not automatically detected landmarks.
        """

        landmark_image = image.copy()
        drawing = ImageDraw.Draw(landmark_image)

        width, height = landmark_image.size

        # Approximate landmark positions for educational illustration
        landmark_positions = [
            # Left eyebrow
            (0.30, 0.33),
            (0.35, 0.31),
            (0.40, 0.32),

            # Right eyebrow
            (0.60, 0.32),
            (0.65, 0.31),
            (0.70, 0.33),

            # Left eye
            (0.33, 0.39),
            (0.37, 0.38),
            (0.41, 0.39),

            # Right eye
            (0.59, 0.39),
            (0.63, 0.38),
            (0.67, 0.39),

            # Nose
            (0.50, 0.42),
            (0.49, 0.48),
            (0.48, 0.54),
            (0.52, 0.54),

            # Mouth
            (0.40, 0.64),
            (0.45, 0.62),
            (0.50, 0.63),
            (0.55, 0.62),
            (0.60, 0.64),
            (0.50, 0.67),

            # Jawline
            (0.28, 0.48),
            (0.27, 0.57),
            (0.30, 0.68),
            (0.36, 0.77),
            (0.43, 0.82),
            (0.50, 0.84),
            (0.57, 0.82),
            (0.64, 0.77),
            (0.70, 0.68),
            (0.73, 0.57),
            (0.72, 0.48)
        ]

        radius = max(3, int(min(width, height) * 0.008))

        pixel_landmarks = []

        for x_ratio, y_ratio in landmark_positions:

            x_position = int(width * x_ratio)
            y_position = int(height * y_ratio)

            pixel_landmarks.append(
                (x_position, y_position)
            )

            drawing.ellipse(
                [
                    x_position - radius,
                    y_position - radius,
                    x_position + radius,
                    y_position + radius
                ],
                fill="red",
                outline="white"
            )

        # Draw simple connections between nearby landmark groups
        connection_groups = [
            pixel_landmarks[0:3],
            pixel_landmarks[3:6],
            pixel_landmarks[6:9],
            pixel_landmarks[9:12],
            pixel_landmarks[12:16],
            pixel_landmarks[16:22],
            pixel_landmarks[22:33]
        ]

        for group in connection_groups:
            if len(group) > 1:
                drawing.line(
                    group,
                    fill="yellow",
                    width=max(1, radius // 2)
                )

        return landmark_image, pixel_landmarks


    def generate_simulated_faceprint(image):
        """
        Generate a simplified numerical template from resized grayscale
        pixel values.

        This is an educational image vector, not a real biometric face embedding.
        """

        grayscale_image = image.convert("L")

        # Resize to create 128 numerical values
        small_image = grayscale_image.resize((16, 8))

        pixel_values = np.array(
            small_image,
            dtype=np.float32
        ).flatten()

        # Scale values between 0 and 1
        pixel_values = pixel_values / 255.0

        # Normalise the vector
        vector_norm = np.linalg.norm(pixel_values)

        if vector_norm > 0:
            pixel_values = pixel_values / vector_norm

        return pixel_values


    def calculate_similarity(first_faceprint, second_faceprint):
        """Compare two numerical faceprint vectors."""

        similarity = cosine_similarity(
            first_faceprint.reshape(1, -1),
            second_faceprint.reshape(1, -1)
        )[0][0]

        return float(similarity)


    # -----------------------------------------------------
    # TEMPORARY SESSION DATABASE
    # -----------------------------------------------------

    if "biometric_database" not in st.session_state:
        st.session_state.biometric_database = {}

    enrol_tab, compare_tab, database_tab = st.tabs(
        [
            "1. Enrol Reference Face",
            "2. Compare Face",
            "3. Stored Entries"
        ]
    )

    # =====================================================
    # TAB 1 — ENROLMENT
    # =====================================================

    with enrol_tab:

        st.subheader("Step 1: Upload a Reference Photograph")

        participant_id = st.text_input(
            "Enter a participant name or ID",
            placeholder="Example: Participant 001",
            key="biometric_participant_id"
        )

        reference_file = st.file_uploader(
            "Upload a clear face photograph",
            type=["jpg", "jpeg", "png"],
            key="biometric_reference_file"
        )

        if reference_file is not None:

            try:
                reference_image = prepare_image(reference_file)

                landmark_image, landmark_positions = (
                    draw_simulated_landmarks(reference_image)
                )

                reference_faceprint = generate_simulated_faceprint(
                    reference_image
                )

                image_col1, image_col2 = st.columns(2)

                with image_col1:
                    st.markdown("#### Original Photograph")

                    st.image(
                        reference_image,
                        use_container_width=True
                    )

                with image_col2:
                    st.markdown("#### Illustrative Facial Landmarks")

                    st.image(
                        landmark_image,
                        use_container_width=True
                    )

                st.info("""
                The displayed points illustrate common facial regions such as
                the eyes, eyebrows, nose, mouth and jawline. They are positioned
                for educational demonstration and are not automatically detected.
                """)

                st.write("---")
                st.subheader("Step 2: Generate a Simplified Faceprint")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Illustrative landmarks",
                    len(landmark_positions)
                )

                metric2.metric(
                    "Faceprint values",
                    len(reference_faceprint)
                )

                metric3.metric(
                    "Storage type",
                    "Session memory"
                )

                st.write("""
                The uploaded image is converted to grayscale and resized. Its
                numerical pixel values are normalised to create a simplified
                128-value faceprint.
                """)

                preview_count = 24

                faceprint_preview = pd.DataFrame(
                    {
                        "Position": range(1, preview_count + 1),
                        "Faceprint value":
                            reference_faceprint[:preview_count]
                    }
                )

                st.dataframe(
                    faceprint_preview,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Only the first 24 values are displayed. The complete "
                    "demonstration faceprint contains 128 values."
                )

                if st.button(
                    "Store Reference Faceprint",
                    type="primary",
                    key="store_biometric_faceprint"
                ):

                    cleaned_id = participant_id.strip()

                    if not cleaned_id:
                        st.error(
                            "Enter a participant name or ID before storing "
                            "the faceprint."
                        )

                    else:
                        st.session_state.biometric_database[
                            cleaned_id
                        ] = reference_faceprint

                        st.success(
                            f"The simplified faceprint for {cleaned_id} "
                            "was stored temporarily."
                        )

            except Exception as error:
                st.error(
                    "The image could not be processed. Upload a valid JPG, "
                    "JPEG or PNG image."
                )

                st.caption(
                    f"Technical detail: {error}"
                )

    # =====================================================
    # TAB 2 — COMPARISON
    # =====================================================

    with compare_tab:

        st.subheader("Step 3: Upload a Photograph for Comparison")

        st.write("""
        The comparison photograph is converted into another numerical vector.
        The new vector is compared with the temporarily stored reference entries.
        """)

        if not st.session_state.biometric_database:
            st.info(
                "No reference faceprints are stored. Enrol a reference image first."
            )

        comparison_file = st.file_uploader(
            "Upload the comparison photograph",
            type=["jpg", "jpeg", "png"],
            key="biometric_comparison_file"
        )

        if comparison_file is not None:

            try:
                comparison_image = prepare_image(
                    comparison_file
                )

                comparison_landmark_image, comparison_landmarks = (
                    draw_simulated_landmarks(comparison_image)
                )

                comparison_faceprint = (
                    generate_simulated_faceprint(
                        comparison_image
                    )
                )

                image_col1, image_col2 = st.columns(2)

                with image_col1:
                    st.markdown("#### Comparison Photograph")

                    st.image(
                        comparison_image,
                        use_container_width=True
                    )

                with image_col2:
                    st.markdown("#### Illustrative Landmarks")

                    st.image(
                        comparison_landmark_image,
                        use_container_width=True
                    )

                if st.session_state.biometric_database:

                    comparison_results = []

                    for stored_id, stored_faceprint in (
                        st.session_state.biometric_database.items()
                    ):

                        similarity = calculate_similarity(
                            comparison_faceprint,
                            stored_faceprint
                        )

                        comparison_results.append(
                            {
                                "Stored Entry": stored_id,
                                "Similarity Score": similarity,
                                "Similarity Percentage":
                                    round(similarity * 100, 2)
                            }
                        )

                    results_df = pd.DataFrame(
                        comparison_results
                    ).sort_values(
                        by="Similarity Score",
                        ascending=False
                    )

                    best_result = results_df.iloc[0]

                    closest_entry = best_result[
                        "Stored Entry"
                    ]

                    similarity_percentage = float(
                        best_result["Similarity Percentage"]
                    )

                    st.write("---")
                    st.subheader("Step 4: Comparison Result")

                    result_col1, result_col2 = st.columns(2)

                    result_col1.metric(
                        "Closest Stored Entry",
                        closest_entry
                    )

                    result_col2.metric(
                        "Similarity",
                        f"{similarity_percentage:.2f}%"
                    )

                    st.progress(
                        min(
                            max(int(similarity_percentage), 0),
                            100
                        )
                    )

                    demonstration_threshold = 95.0

                    if similarity_percentage >= demonstration_threshold:
                        st.success(
                            "Demonstration result: MATCH"
                        )
                    else:
                        st.error(
                            "Demonstration result: NO MATCH"
                        )

                    st.dataframe(
                        results_df[
                            [
                                "Stored Entry",
                                "Similarity Percentage"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption("""
                    The 95% threshold is used only to demonstrate how a matching
                    decision can be made. It is not a validated biometric threshold.
                    """)

            except Exception as error:
                st.error(
                    "The comparison image could not be processed."
                )

                st.caption(
                    f"Technical detail: {error}"
                )

    # =====================================================
    # TAB 3 — DATABASE
    # =====================================================

    with database_tab:

        st.subheader("Temporary Demonstration Database")

        st.write("""
        Stored entries remain only in the current Streamlit session. They are
        not written to the survey dataset or to a permanent biometric database.
        """)

        if st.session_state.biometric_database:

            database_summary = pd.DataFrame(
                [
                    {
                        "Participant ID": participant_id,
                        "Faceprint Length": len(faceprint),
                        "Storage": "Temporary session memory"
                    }
                    for participant_id, faceprint
                    in st.session_state.biometric_database.items()
                ]
            )

            st.dataframe(
                database_summary,
                use_container_width=True,
                hide_index=True
            )

            selected_entry = st.selectbox(
                "Select an entry to preview",
                options=list(
                    st.session_state.biometric_database.keys()
                ),
                key="selected_biometric_entry"
            )

            selected_faceprint = (
                st.session_state.biometric_database[
                    selected_entry
                ]
            )

            selected_preview = pd.DataFrame(
                {
                    "Position": range(1, 21),
                    "Stored value": selected_faceprint[:20]
                }
            )

            st.dataframe(
                selected_preview,
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "Clear Temporary Database",
                key="clear_biometric_database"
            ):

                st.session_state.biometric_database = {}

                st.success(
                    "The temporary biometric database was cleared."
                )

                st.rerun()

        else:
            st.info(
                "No faceprints are currently stored."
            )

    st.write("---")
    st.subheader("How Real Facial Identification Works")

    st.markdown("""
    **1. Image capture:** A camera captures the user's face.

    **2. Face detection:** The system identifies the facial region in the image.

    **3. Landmark extraction:** Key facial points around the eyes, nose, mouth
    and face shape are located.

    **4. Feature extraction:** A trained model converts facial characteristics
    into a biometric template or embedding.

    **5. Secure storage:** The protected template is stored in an encrypted
    database.

    **6. Comparison:** A new template is compared with stored templates.

    **7. Authentication decision:** Access is allowed or denied according to a
    tested similarity threshold.
    """)

    st.subheader("Privacy and Security Considerations")

    st.write("""
    Real banking systems require encrypted biometric templates, secure storage,
    access controls, liveness detection, presentation-attack protection, tested
    matching thresholds and clear data-retention policies. This page illustrates
    the workflow only and does not implement genuine biometric authentication.
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
