import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/random_forest_model.pkl")


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

file_path = "data/crx.data"

df = pd.read_csv(file_path, header=None)

columns = [
    "Gender",
    "Age",
    "Debt",
    "Married",
    "BankCustomer",
    "Education",
    "Ethnicity",
    "YearsEmployed",
    "PriorDefault",
    "Employed",
    "CreditScore",
    "DriversLicense",
    "Citizen",
    "ZipCode",
    "Income",
    "Approved"
]

df.columns = columns


# ============================================================
# PREPROCESS DATA
# ============================================================

df = df.replace("?", pd.NA)

# Age
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].median())


# Fill missing categorical values
categorical_columns_for_missing = [
    "Gender",
    "Married",
    "BankCustomer",
    "Education",
    "Ethnicity",
    "ZipCode"
]

for column in categorical_columns_for_missing:
    df[column] = df[column].fillna(df[column].mode()[0])


# ZipCode must be numeric
df["ZipCode"] = pd.to_numeric(df["ZipCode"])


# ============================================================
# CREATE ENCODERS
# ============================================================

categorical_columns = [
    "Gender",
    "Married",
    "BankCustomer",
    "Education",
    "Ethnicity",
    "PriorDefault",
    "Employed",
    "DriversLicense",
    "Citizen",
    "Approved"
]

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    encoders[column] = encoder


# ============================================================
# STREAMLIT TITLE
# ============================================================

st.title("💳 Credit Card Approval Prediction")

st.write(
    "Enter the applicant details below to predict whether "
    "the credit card application will be approved or rejected."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_options = {
        "Male": "a",
        "Female": "b"
    }

    gender = st.selectbox(
        "Gender",
        list(gender_options.keys())
    )


    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )


    # --------------------------------------------------------
    # DEBT
    # --------------------------------------------------------

    debt = st.number_input(
        "Debt",
        min_value=0.0,
        value=5.0,
        step=0.1
    )


    # --------------------------------------------------------
    # MARITAL STATUS
    # --------------------------------------------------------

    married_options = {
        "Married": "u",
        "Single": "y",
        "Other": "l"
    }

    married = st.selectbox(
        "Marital Status",
        list(married_options.keys())
    )


    # --------------------------------------------------------
    # BANK CUSTOMER
    # --------------------------------------------------------

    bank_customer_options = {
        "Existing Bank Customer": "g",
        "Other Bank Customer": "p",
        "Not a Bank Customer": "gg"
    }

    bank_customer = st.selectbox(
        "Bank Customer",
        list(bank_customer_options.keys())
    )


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    education_options = {
        "College / University": "c",
        "High School": "d",
        "Graduate": "cc",
        "Postgraduate": "i",
        "Vocational Education": "k",
        "Other": "j",
        "Unknown": "m"
    }

    education = st.selectbox(
        "Education",
        list(education_options.keys())
    )


    # --------------------------------------------------------
    # ETHNICITY
    # --------------------------------------------------------

    ethnicity_options = {
        "White": "v",
        "Black": "h",
        "Asian": "bb",
        "Other": "z",
        "Unknown": "dd",
        "Hispanic": "j",
        "Native American": "n"
    }

    ethnicity = st.selectbox(
        "Ethnicity",
        list(ethnicity_options.keys())
    )


with col2:

    # --------------------------------------------------------
    # YEARS EMPLOYED
    # --------------------------------------------------------

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )


    # --------------------------------------------------------
    # PRIOR DEFAULT
    # --------------------------------------------------------

    prior_default = st.selectbox(
        "Previous Credit Default",
        ["Yes", "No"]
    )


    # --------------------------------------------------------
    # EMPLOYED
    # --------------------------------------------------------

    employed = st.selectbox(
        "Currently Employed",
        ["Yes", "No"]
    )


    # --------------------------------------------------------
    # CREDIT SCORE
    # --------------------------------------------------------

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


    # --------------------------------------------------------
    # DRIVER'S LICENSE
    # --------------------------------------------------------

    drivers_license = st.selectbox(
        "Driver's License",
        ["Yes", "No"]
    )


    # --------------------------------------------------------
    # CITIZENSHIP
    # --------------------------------------------------------

    citizen_options = {
        "Citizen by Birth": "g",
        "Citizen by Naturalization": "p",
        "Other": "s"
    }

    citizen = st.selectbox(
        "Citizenship",
        list(citizen_options.keys())
    )


    # --------------------------------------------------------
    # ZIP CODE
    # --------------------------------------------------------

    zip_code = st.number_input(
        "ZIP Code",
        min_value=0,
        max_value=99999,
        value=100,
        step=1
    )


    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    income = st.number_input(
        "Income",
        min_value=0,
        max_value=1000000,
        value=5000,
        step=100
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Credit Card Approval",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Convert Yes / No into original dataset values

    prior_default_code = (
        "t" if prior_default == "Yes" else "f"
    )

    employed_code = (
        "t" if employed == "Yes" else "f"
    )

    drivers_license_code = (
        "t" if drivers_license == "Yes" else "f"
    )


    # --------------------------------------------------------
    # CREATE RAW INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "Gender": gender_options[gender],

        "Age": age,

        "Debt": debt,

        "Married": married_options[married],

        "BankCustomer": bank_customer_options[bank_customer],

        "Education": education_options[education],

        "Ethnicity": ethnicity_options[ethnicity],

        "YearsEmployed": years_employed,

        "PriorDefault": prior_default_code,

        "Employed": employed_code,

        "CreditScore": credit_score,

        "DriversLicense": drivers_license_code,

        "Citizen": citizen_options[citizen],

        "ZipCode": zip_code,

        "Income": income

    }])


    # --------------------------------------------------------
    # ENCODE INPUT DATA
    # --------------------------------------------------------

    input_encoded = input_data.copy()

    for column in categorical_columns:

        if column != "Approved":

            input_encoded[column] = encoders[column].transform(
                input_encoded[column].astype(str)
            )


    # --------------------------------------------------------
    # ENSURE SAME COLUMN ORDER
    # --------------------------------------------------------

    input_encoded = input_encoded[
        [
            "Gender",
            "Age",
            "Debt",
            "Married",
            "BankCustomer",
            "Education",
            "Ethnicity",
            "YearsEmployed",
            "PriorDefault",
            "Employed",
            "CreditScore",
            "DriversLicense",
            "Citizen",
            "ZipCode",
            "Income"
        ]
    ]


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_encoded)[0]

    probabilities = model.predict_proba(input_encoded)[0]


    # ========================================================
    # IMPORTANT TARGET MAPPING
    # ========================================================
    #
    # LabelEncoder sorts:
    #
    # "+" -> 0
    # "-" -> 1
    #
    # Therefore:
    #
    # 0 = APPROVED
    # 1 = REJECTED
    #
    # ========================================================

    if prediction == 0:

        result = "APPROVED"

        st.success(
            "✅ Credit Card Application APPROVED"
        )

    else:

        result = "REJECTED"

        st.error(
            "❌ Credit Card Application REJECTED"
        )


    # --------------------------------------------------------
    # SHOW PROBABILITY
    # --------------------------------------------------------

    approved_probability = probabilities[0] * 100
    rejected_probability = probabilities[1] * 100


    st.subheader("Prediction Probability")

    probability_col1, probability_col2 = st.columns(2)


    with probability_col1:

        st.metric(
            "Approval Probability",
            f"{approved_probability:.2f}%"
        )


    with probability_col2:

        st.metric(
            "Rejection Probability",
            f"{rejected_probability:.2f}%"
        )


    # --------------------------------------------------------
    # APPLICANT DETAILS
    # --------------------------------------------------------

    with st.expander("📋 View Applicant Details"):

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Debt:** {debt}"
        )

        st.write(
            f"**Marital Status:** {married}"
        )

        st.write(
            f"**Bank Customer:** {bank_customer}"
        )

        st.write(
            f"**Education:** {education}"
        )

        st.write(
            f"**Ethnicity:** {ethnicity}"
        )

        st.write(
            f"**Years Employed:** {years_employed}"
        )

        st.write(
            f"**Previous Credit Default:** {prior_default}"
        )

        st.write(
            f"**Currently Employed:** {employed}"
        )

        st.write(
            f"**Credit Score:** {credit_score}"
        )

        st.write(
            f"**Driver's License:** {drivers_license}"
        )

        st.write(
            f"**Citizenship:** {citizen}"
        )

        st.write(
            f"**ZIP Code:** {zip_code}"
        )

        st.write(
            f"**Income:** {income}"
        )
