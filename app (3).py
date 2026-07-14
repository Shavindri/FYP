import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    option = st.selectbox(
        "Select Analysis",
        [
            "Age Distribution",
            "Gender Distribution",
            "Banking Usage",
            "Platform Usage",
            "Awareness Level",
            "Behaviour Level",
            "Biometric Level",
            "Correlation Heatmap"
        ]
    )

    if option == "Age Distribution":
        st.bar_chart(df["Age"].value_counts())

    elif option == "Gender Distribution":
        st.bar_chart(df["Gender"].value_counts())

    elif option == "Banking Usage":
        st.bar_chart(df["Banking_Use"].value_counts())

    elif option == "Platform Usage":
        st.bar_chart(df["Platform"].value_counts())

    elif option == "Awareness Level":
        st.bar_chart(df["Awareness_Level"].value_counts())

    elif option == "Behaviour Level":
        st.bar_chart(df["Behaviour_Level"].value_counts())

    elif option == "Biometric Level":
        st.bar_chart(df["Biometric_Level"].value_counts())

    elif option == "Correlation Heatmap":
        corr_cols = [
            "Awareness_Score",
            "Opinion_Score",
            "Behaviour_Score",
            "Biometric_Score"
        ]

        corr = df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)


elif page == "Cyber Awareness Assessment":
    st.title("Cyber Awareness Assessment")

    st.write("""
    Answer the following questions. The application will calculate your cyber awareness score
    and provide personalised feedback.
    """)

    score = 0
    total = 8
    feedback = []

    phishing = st.selectbox(
        "1. What would you do if you received an email claiming your bank account will be suspended unless you click a link?",
        [
            "Verify the request using the bank's official website or customer service",
            "Ignore it completely",
            "Click the link immediately"
        ]
    )

    if phishing == "Verify the request using the bank's official website or customer service":
        score += 1
    else:
        feedback.append("Improve phishing awareness by verifying suspicious emails through official bank channels.")

    otp = st.selectbox(
        "2. What would you do if someone claiming to be from your bank asks for your OTP?",
        [
            "End the call and contact the bank through official channels",
            "Ask why it is needed",
            "Ignore the request but continue the conversation",
            "Share the OTP"
        ]
    )

    if otp == "End the call and contact the bank through official channels":
        score += 1
    else:
        feedback.append("Never share OTPs. Banks will not ask customers to share OTPs, passwords, or PINs.")

    password = st.selectbox(
        "3. How often do you use a unique password for online banking?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if password == "Always":
        score += 1
    else:
        feedback.append("Improve password practices by using a strong and unique password for online banking.")

    mfa = st.selectbox(
        "4. How often do you enable Multi-Factor Authentication when it is available?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if mfa == "Always":
        score += 1
    else:
        feedback.append("Enable Multi-Factor Authentication to add an extra layer of protection.")

    website = st.selectbox(
        "5. How often do you verify the banking website before entering login details?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if website == "Always":
        score += 1
    else:
        feedback.append("Always check website authenticity before entering banking credentials.")

    public_wifi = st.selectbox(
        "6. Would you use public Wi-Fi for online banking?",
        ["Never", "Sometimes", "Often"]
    )

    if public_wifi == "Never":
        score += 1
    else:
        feedback.append("Avoid public Wi-Fi for online banking because it may expose sensitive information.")

    biometric = st.selectbox(
        "7. Do you understand how biometric authentication protects online banking accounts?",
        ["Yes", "Somewhat", "No"]
    )

    if biometric == "Yes":
        score += 1
    else:
        feedback.append("Improve understanding of biometric authentication and how biometric data is protected.")

    ai_scams = st.selectbox(
        "8. What would you do if you received a voice call that sounded like your bank manager asking you to approve a transaction?",
        [
            "Verify the request using the bank's official contact details",
            "End the call without checking",
            "Approve the transaction"
        ]
    )

    if ai_scams == "Verify the request using the bank's official contact details":
        score += 1
    else:
        feedback.append("Be cautious of AI-generated scams and deepfake calls. Always verify urgent requests officially.")

    if st.button("Calculate Awareness Score"):
        percentage = round((score / total) * 100, 2)

        st.subheader("Your Cyber Awareness Score")
        st.write(f"{percentage}%")

        if percentage >= 75:
            st.success("Awareness Level: High")
        elif percentage >= 50:
            st.warning("Awareness Level: Moderate")
        else:
            st.error("Awareness Level: Low")

        st.subheader("Personalised Feedback")

        if feedback:
            for item in feedback:
                st.write("- " + item)
        else:
            st.write("You demonstrated strong cyber awareness across all assessed areas.")


elif page == "Biometric Training Demo":
    st.title("Biometric Facial Identification Training Demonstration")

    st.write("""
    This section explains how biometric facial identification works in a simplified way.
    It is not a real authentication system and does not store or verify biometric data.
    """)

    uploaded_image = st.file_uploader(
        "Upload a face image for demonstration purposes",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        st.subheader("Step 1: Face Detection")
        st.write("""
        A biometric system first detects whether a face is present in the image.
        """)

        st.subheader("Step 2: Facial Landmark Identification")
        st.write("""
        The system identifies key facial landmarks such as the eyes, nose, mouth, jawline,
        and distance between facial features.
        """)

        st.subheader("Step 3: Faceprint Generation")
        st.write("""
        These facial measurements are converted into a mathematical representation known as a faceprint.
        A faceprint is not the same as a normal image; it is a numerical pattern used for comparison.
        """)

        st.subheader("Step 4: Matching Process")
        st.write("""
        In a real authentication system, the generated faceprint would be compared with a stored template
        to decide whether access should be granted.
        """)

        st.warning("""
        This demonstration is educational only. It does not authenticate users, store biometric data,
        or perform real facial recognition.
        """)

        st.subheader("Privacy and Security Considerations")
        st.write("""
        Biometric systems must protect user privacy by securely storing biometric templates,
        limiting access to sensitive data, and clearly explaining how biometric information is used.
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
