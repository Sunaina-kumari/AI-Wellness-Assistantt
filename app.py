import streamlit as st
import pandas as pd
import requests
import json
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="AI Wellness Assistant",

    page_icon="🤖",

    layout="centered"
)


# =========================================================
st.markdown("""

<style>

.stApp {

    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );

    color: white;
}


/* MAIN TITLE */

h1 {

    color: white !important;

    text-align: center;

    font-size: 42px !important;

    font-weight: bold;
}


/* SUBTEXT */

.small-text {

    color: #e2e8f0;

    font-size: 18px;

    text-align: center;
            label {

    color: white !important;

    font-size: 16px !important;

    font-weight: 500;
}
}


p, span, div {

    color: white;
}




            
label {

    color: #f8fafc !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    opacity: 1 !important;

    letter-spacing: 0.3px;
}


/* CHAT MESSAGES */

[data-testid="stChatMessage"] {

    background-color: rgba(255,255,255,0.10);

    border-radius: 15px;

    padding: 15px;

    margin-bottom: 15px;

    border: 1px solid rgba(255,255,255,0.15);
}


/* CHATBOT TEXT */

[data-testid="stChatMessage"] p {

    color: white !important;

    font-size: 17px;

    line-height: 1.7;
}
/* INPUT BOX */

.stTextInput input {

    background-color: white !important;

    color: black !important;

    border-radius: 12px;

    padding: 12px;

    font-size: 16px;
}


/* SELECTBOX VALUE */

/* FORCE SELECTED TEXT COLOR */

.stSelectbox * {
    color: black !important;
}
            

.stSelectbox div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* DROPDOWN MENU CONTAINER */

div[data-baseweb="popover"] {
    background-color: white !important;
}

/* DROPDOWN OPTIONS */

li[role="option"] {
    background-color: white !important;
    color: black !important;
    font-weight: bold !important;
}

/* OPTION TEXT */

li[role="option"] * {
    color: black !important;
}

/* HOVER */

li[role="option"]:hover {
    background-color: #dbeafe !important;
    color: black !important;
}



            
/* BUTTON */

.stButton button {

    background: linear-gradient(
        to right,
        #2563eb,
        #3b82f6
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 10px 25px;

    font-size: 16px;

    font-weight: bold;

    width: 100%;
}


.stButton button:hover {

    background: linear-gradient(
        to right,
        #1d4ed8,
        #2563eb
    );
}


/* RESULT CARD */
            /* METRIC TEXT */

[data-testid="stMetric"] {

    background-color: rgba(255,255,255,0.08);

    padding: 15px;

    border-radius: 15px;

    text-align: center;
}


[data-testid="stMetricLabel"] {

    color: white !important;

    font-size: 18px !important;
}


[data-testid="stMetricValue"] {

    color: white !important;

    font-size: 38px !important;

    font-weight: bold !important;
}

.result-card {

    background: linear-gradient(
        to right,
        #2563eb,
        #1d4ed8
    );

    padding: 25px;

    border-radius: 20px;

    text-align: center;

    color: white;

    font-size: 24px;

    font-weight: bold;

    margin-top: 20px;
}

</style>

""", unsafe_allow_html=True)
# =========================================================
# OPENROUTER API
# =========================================================

OPENROUTER_API_KEY = ""


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "Indian_Kids_Screen_Time.csv"
)
df["Health_Impacts"] = (
    df["Health_Impacts"]
    .fillna("none")
    .astype(str)
    .str.lower()
)

df.columns = df.columns.str.strip()

df.rename(columns={
    "Avg_Daily_Screen_Time_hr": "Screen_Time"
}, inplace=True)

df = df.ffill()


# =========================================================
# TEXT NORMALIZATION
# =========================================================

text_columns = df.select_dtypes(
    include="object"
).columns

for col in text_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# =========================================================

# =========================================================
# CREATE RISK LEVEL
# =========================================================

def create_risk_level(row):

    risk_score = 0


    # =====================================================
    # SCREEN TIME
    # =====================================================

    if row["Screen_Time"] >= 8:

        risk_score += 4


    elif row["Screen_Time"] >= 6:

        risk_score += 3


    elif row["Screen_Time"] >= 4:

        risk_score += 2


    else:

        risk_score += 1


    # =====================================================
    # EDUCATIONAL / RECREATIONAL RATIO
    # =====================================================

    ratio = row["Educational_to_Recreational_Ratio"]


    # Mostly recreational usage

    if ratio < 0.30:

        risk_score += 3


    elif ratio < 0.50:

        risk_score += 2


    # Balanced usage

    elif ratio < 1.5:

        risk_score += 1


    # Educational usage higher than recreational

    else:

        risk_score -= 2


    # =====================================================
    # HEALTH IMPACTS
    # =====================================================

    impacts = row["Health_Impacts"]


    if "poor sleep" in impacts:

        risk_score += 2


    if "Vision strain" in impacts:

        risk_score += 2


    if "anxiety" in impacts:

        risk_score += 2


    if "torpid risk" in impacts:

        risk_score += 2


    # =====================================================
    # DEVICE TYPE
    # =====================================================

    device = row["Primary_Device"]


    if device == "smartphone":

        risk_score += 2


    elif device == "tablet":

        risk_score += 1


    elif device == "tv":

        risk_score += 1


    # =====================================================
   
    # =====================================================
    # FINAL RISK CLASSIFICATION
    # =====================================================

    if risk_score <= 5:

        return "safe"


    elif risk_score <= 9:

        return "moderate"


    elif risk_score <= 13:

        return "high"


    else:

        return "critical"






# =========================================================
# CREATE RISK LEVEL COLUMN
# =========================================================

df["Risk_Level"] = df.apply(
    create_risk_level,
    axis=1
)


# =========================================================
# MACHINE LEARNING
# =========================================================

df_ml = df.copy()

categorical_columns = df_ml.select_dtypes(
    include="object"
).columns

label_encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    df_ml[col] = encoder.fit_transform(
        df_ml[col]
    )

    label_encoders[col] = encoder

print("Health Impact Classes:")
print(label_encoders["Health_Impacts"].classes_)




features = [

    "Age",

    "Screen_Time",

    "Educational_to_Recreational_Ratio",

    "Gender",

    "Primary_Device",

    "Health_Impacts",

    "Urban_or_Rural"

]

target = "Risk_Level"

X = df_ml[features]

y = df_ml[target]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

model.fit(X_train, y_train)




# ============================================
# MODEL EVALUATION
# ============================================

y_pred = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred))

#########table2

results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],

    "Value": [
        round(accuracy,4),
        round(precision_score(y_test, y_pred, average="weighted"),4),
        round(recall_score(y_test, y_pred, average="weighted"),4),
        round(f1_score(y_test, y_pred, average="weighted"),4)
    ]

})

print("\nMODEL PERFORMANCE TABLE\n")
print(results)

####table3

risk_table = pd.DataFrame(
    df["Risk_Level"].value_counts()
)

risk_table.columns = ["Count"]

print("\nRISK LEVEL DISTRIBUTION TABLE\n")
print(risk_table)

# =========================================================
# API FUNCTION
# =========================================================

def ask_ai(prompt):

    url = "https://openrouter.ai/api/v1/chat/completions"


    headers = {

        "Authorization": f"Bearer {OPENROUTER_API_KEY}",

        "Content-Type": "application/json"

    }


    data = {

        "model": "openai/gpt-3.5-turbo",

        "messages": [

            {

                "role": "user",

                "content": prompt

            }

        ]

    }


    response = requests.post(

        url,

        headers=headers,

        data=json.dumps(data)

    )


    result = response.json()


    if "choices" in result:

        return result["choices"][0]["message"]["content"]

    else:
        return f"AI response unavailable: {result}"

# =========================================================
# TITLE
# =========================================================

st.markdown("""

<h1 style='text-align:center;'>
🤖 AI Conversational Wellness Assistant
</h1>

<p class='small-text' style='text-align:center;'>

AI-powered child digital wellness assessment
using conversational interaction and machine learning.

</p>

""", unsafe_allow_html=True)


# =========================================================

# =========================================================
# INPUT VALIDATION HELPERS
# =========================================================

def validate_time_inputs(screen_time, educational_time, recreational_time):
    """Validate consistency of total, educational, and recreational screen time."""

    if screen_time < 0 or educational_time < 0 or recreational_time < 0:
        return False, "Time values cannot be negative."

    if screen_time > 24:
        return False, "Daily screen time cannot be greater than 24 hours."

    if educational_time > screen_time:
        return False, (
            f"Educational time ({educational_time:g} hrs) cannot be greater "
            f"than total screen time ({screen_time:g} hrs)."
        )

    if recreational_time > screen_time:
        return False, (
            f"Recreational time ({recreational_time:g} hrs) cannot be greater "
            f"than total screen time ({screen_time:g} hrs)."
        )

    combined_time = educational_time + recreational_time

    if combined_time > screen_time:
        return False, (
            f"Invalid input: Educational time ({educational_time:g} hrs) + "
            f"Recreational time ({recreational_time:g} hrs) = "
            f"{combined_time:g} hrs, which is greater than total daily "
            f"screen time ({screen_time:g} hrs). Please enter valid values."
        )

    return True, ""


# =========================================================
# SESSION STATE
# =========================================================

if "history_saved" not in st.session_state:
    st.session_state.history_saved = False

if "step" not in st.session_state:

    st.session_state.step = 1


if "answers" not in st.session_state:

    st.session_state.answers = {}


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "current_ai_message" not in st.session_state:

    st.session_state.current_ai_message = ""


# =========================================================
# CHAT HISTORY DISPLAY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================

# =========================================================
# STEP 1 — AGE
# =========================================================

if st.session_state.step == 1:


    if st.session_state.current_ai_message == "":

        st.session_state.current_ai_message = ask_ai(

            """
            Generate a short friendly chatbot message
            that welcomes the user AND asks:

            "What is the child's age?"

            Keep response short and conversational.
            """
        )


    ai_message = st.session_state.current_ai_message


    st.chat_message("assistant").write(
        ai_message
    )


    age = st.text_input(

        "Enter Child Age",

        value=str(
            st.session_state.answers.get(
                "Age",
                ""
            )
        ),

        placeholder="Example: 12"
    )


    if st.button("Next"):
        
        
        if st.session_state.step == 1:
            st.session_state.history_saved = False

        st.session_state.answers["Age"] = int(age)

        st.session_state.current_ai_message = ""

        st.session_state.step = 2

        st.rerun()
   

# =========================================================
# STEP 2 — SCREEN TIME
# =========================================================

elif st.session_state.step == 2:

    ai_message = ask_ai(

        """
        Generate a short conversational message
        asking daily screen time.
        """
    )

    st.chat_message("assistant").write(
        ai_message
    )


    screen_time = st.text_input(

        "Daily Screen Time",

        value=str(
            st.session_state.answers.get(
                "Screen_Time",
                ""
            )
        ),

        placeholder="Example: 5"
    )

    st.caption(
        "Enter the child's actual average daily screen time. "
        "This value is not treated as a fixed 6-hour limit."
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 1

            st.rerun()


    with col2:

        if st.button("Next"):

            st.session_state.answers[
                "Screen_Time"
            ] = float(screen_time)

            st.session_state.step = 3

            st.rerun()


# =========================================================
# STEP 3 — EDUCATIONAL TIME
# =========================================================

elif st.session_state.step == 3:

    st.chat_message("assistant").write(

        "📚 How many hours are spent on online study or learning daily?"
    )


    educational_time = st.text_input(

        "Educational Time",

        value=str(
            st.session_state.answers.get(
                "Educational_Time",
                ""
            )
        ),

        placeholder="Example: 2"
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 2

            st.rerun()


    with col2:

        if st.button("Next"):

            try:
                educational_value = float(educational_time)
            except ValueError:
                st.error("Please enter a valid number for educational time.")
                st.stop()

            if educational_value < 0:
                st.error("Educational time cannot be negative. Please enter a valid value.")
                st.stop()

            screen_value = float(
                st.session_state.answers.get("Screen_Time", 0)
            )

            if educational_value > screen_value:
                st.error(
                    f"Educational time ({educational_value:g} hrs) cannot be "
                    f"greater than total screen time ({screen_value:g} hrs). "
                    "Please re-enter a valid value."
                )
                st.stop()

            st.session_state.answers[
                "Educational_Time"
            ] = educational_value

            st.session_state.step = 4

            st.rerun()


# =========================================================
# STEP 4 — RECREATIONAL TIME
# =========================================================

elif st.session_state.step == 4:

    st.chat_message("assistant").write(

        "🎮 How many hours are spent on games, videos, or social media daily?"
    )

    st.caption(
        "Note: Educational time + recreational time must not exceed "
        "the child's total daily screen time."
    )


    recreational_time = st.text_input(

        "Recreational Time",

        value=str(
            st.session_state.answers.get(
                "Recreational_Time",
                ""
            )
        ),

        placeholder="Example: 4"
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 3

            st.rerun()


    with col2:

        if st.button("Next"):

            try:
                recreational_value = float(recreational_time)
            except ValueError:
                st.error("Please enter a valid number for recreational time.")
                st.stop()

            screen_value = float(
                st.session_state.answers.get("Screen_Time", 0)
            )

            educational_value = float(
                st.session_state.answers.get("Educational_Time", 0)
            )

            is_valid, validation_message = validate_time_inputs(
                screen_value,
                educational_value,
                recreational_value
            )

            if not is_valid:
                st.error(validation_message)
                st.warning(
                    "Please correct the recreational time. "
                    "If the educational time is incorrect, use the Previous button "
                    "to edit it."
                )
                st.stop()

            st.session_state.answers[
                "Recreational_Time"
            ] = recreational_value

            st.session_state.step = 5

            st.rerun()


# =========================================================
# STEP 5 — GENDER
# =========================================================

elif st.session_state.step == 5:

    ai_message = ask_ai("""
    Ask the user politely for the child's gender.
    Keep it short and friendly.
    """)

    st.chat_message("assistant").write(ai_message)


    gender_options = [

        "male",

        "female"
    ]


    gender = st.selectbox(

        "Gender",

        gender_options,

        index=gender_options.index(

            st.session_state.answers.get(
                "Gender",
                "male"
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 4

            st.rerun()


    with col2:

        if st.button("Next"):

            st.session_state.answers[
                "Gender"
            ] = gender

            st.session_state.step = 6

            st.rerun()


# =========================================================
# STEP 6 — DEVICE
# =========================================================

elif st.session_state.step == 6:

    ai_message = ask_ai("""
    Ask which device the child uses most frequently.
    Keep it friendly and conversational.
    """)
    st.chat_message("assistant").write(
    ai_message
    )

    device_options = [

        "smartphone",

        "tablet",

        "tv",

        "laptop"
    ]


    device = st.selectbox(

        "Primary Device",

        device_options,

        index=device_options.index(

            st.session_state.answers.get(
                "Primary_Device",
                "smartphone"
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 5

            st.rerun()


    with col2:

        if st.button("Next"):

            st.session_state.answers[
                "Primary_Device"
            ] = device

            st.session_state.step = 7

            st.rerun()


# =========================================================
# STEP 7 — HEALTH IMPACT
# =========================================================

elif st.session_state.step == 7:

    ai_message = ask_ai("""
    Ask whether the child has experienced health effects from screen usage.
    Keep it short and parent-friendly.
    """)
    st.chat_message("assistant").write(
    ai_message
    )


    health_options = [

        "none",

        "poor sleep",

        "Vision strain",

        "anxiety",

        "torpid risk"
    ]


    health = st.selectbox(

        "Health Impact",

        health_options,

        index=health_options.index(

            st.session_state.answers.get(
                "Health_Impacts",
                "none"
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 6

            st.rerun()


    with col2:

        if st.button("Next"):

            st.session_state.answers[
                "Health_Impacts"
            ] = health

            st.session_state.step = 8

            st.rerun()


# =========================================================
# STEP 8 — LOCATION
# =========================================================

elif st.session_state.step == 8:

    st.chat_message("assistant").write(

        "🏠 Where does the child primarily live?"
    )


    location_options = [

        "urban",

        "rural"
    ]


    location = st.selectbox(

        "Location",

        location_options,

        index=location_options.index(

            st.session_state.answers.get(
                "Urban_or_Rural",
                "urban"
            )
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ Previous"):

            st.session_state.step = 7

            st.rerun()


    with col2:

        if st.button("Analyze Wellness Risk"):

            st.session_state.answers[
                "Urban_or_Rural"
            ] = location

            st.session_state.step = 10

            st.rerun()


# =====================================================



# =====================================================

# =========================================================


# =========================================================
# STEP 10 — ML PREDICTION
# =========================================================


if st.session_state.step == 10:

    answers = st.session_state.answers

    # Final safety check before ML prediction.
    # This prevents inconsistent values from ever reaching the model.
    is_valid, validation_message = validate_time_inputs(
        float(answers["Screen_Time"]),
        float(answers["Educational_Time"]),
        float(answers["Recreational_Time"])
    )

    if not is_valid:
        st.error(validation_message)
        st.warning(
            "The assessment cannot continue until the time values are corrected."
        )

        if st.button("⬅ Edit Time Inputs"):
            st.session_state.step = 4
            st.rerun()

        st.stop()

    ratio = (
        answers["Educational_Time"] /
        max(answers["Recreational_Time"], 1)
    )

    sample_df = pd.DataFrame([{

        "Age": answers["Age"],

        "Screen_Time": answers["Screen_Time"],

        "Educational_to_Recreational_Ratio": ratio,

        "Gender": answers["Gender"],

        "Primary_Device": answers["Primary_Device"],

        "Health_Impacts": answers["Health_Impacts"],

        "Urban_or_Rural": answers["Urban_or_Rural"]

    }])

    sample_encoded = sample_df.copy()

    for col in categorical_columns:

        if col != "Risk_Level":

            sample_encoded[col] = (
                label_encoders[col]
                .transform(sample_encoded[col])
            )

    prediction = model.predict(
        sample_encoded[features]
    )[0]

    final_prediction = (
        label_encoders["Risk_Level"]
        .inverse_transform([prediction])[0]
    )

    history = pd.DataFrame([{
        "Date": pd.Timestamp.now(),
        "Age": answers["Age"],
        "Screen_Time": answers["Screen_Time"],
        "Educational_Time": answers["Educational_Time"],
        "Recreational_Time": answers["Recreational_Time"],
        "Gender": answers["Gender"],
        "Primary_Device": answers["Primary_Device"],
        "Health_Impacts": answers["Health_Impacts"],
        "Urban_or_Rural": answers["Urban_or_Rural"],
        "Risk_Level": final_prediction
    }])
    
   

    if not st.session_state.history_saved:

        history.to_csv(
            "user_history.csv",
            mode="a",
            header=not os.path.exists("user_history.csv"),
            index=False
        )

        st.session_state.history_saved = True

    st.markdown(
        f"""
        <div class="result-card">
        🧠 Predicted Risk Level:
        <br><br>
        {final_prediction.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", answers["Age"])

    with col2:
        st.metric(
            "Screen Time",
            f"{answers['Screen_Time']} hrs"
        )

    with col3:
        st.metric(
            "Risk Level",
            final_prediction.upper()
        )
    guidance = ask_ai(
        f"""
        You are a child digital wellness expert.

        Child Age: {answers['Age']}
        Screen Time: {answers['Screen_Time']} hours
        Risk Level: {final_prediction}

        Give personalized wellness guidance.

        Requirements:
        - 4 to 5 bullet points
        - Simple language
        - Practical advice
        """
        )

    st.markdown("## 🤖 AI Wellness Guidance")

    st.success(guidance)       
   

    course_recommendation = ask_ai(
    f"""
    Recommend 5 educational websites or online courses for a child.

    Child Age: {answers['Age']}
    Risk Level: {final_prediction}

    Requirements:
    - Include website name
    - Include direct website link
    - Give a short description
    - Return in bullet points
    """
    ) 

    st.markdown("## 📚 Recommended Learning Courses")
    st.info(course_recommendation)



# ==========================================
# STREAMLIT APP ENDS
# ==========================================

if __name__ == "__main__":
    pass


# ==========================================
# PAPER VISUALIZATIONS
# ==========================================

import matplotlib.pyplot as plt

risk_counts = df["Risk_Level"].value_counts()

plt.figure(figsize=(8,5))

bars = plt.bar(
    risk_counts.index,
    risk_counts.values
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center"
    )

plt.xlabel("Risk Level")
plt.ylabel("Number of Children")
##=###plt.title("Risk Level Distribution")###

plt.tight_layout()

plt.savefig(
    "risk_level_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(df["Risk_Level"].unique())
print("✅ Risk graph saved successfully")





# =========================================================
# HEALTH IMPACT DISTRIBUTION
# =========================================================

import matplotlib.pyplot as plt

health_counts = (
    df["Health_Impacts"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12,6))

bars = plt.barh(
    health_counts.index,
    health_counts.values
)

plt.xlabel("Number of Children")
plt.ylabel("Health Impacts")
#######==plt.title("Health Impact Distribution")

for bar in bars:
    plt.text(
        bar.get_width() + 20,
        bar.get_y() + bar.get_height()/2,
        str(int(bar.get_width())),
        va="center"
    )

plt.tight_layout()

plt.savefig(
    "health_impact_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✅ Health Impact graph saved successfully")

########======confusion matrix==================================######################

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)

###===plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✅ Confusion Matrix saved successfully")

##############==========pie chart

risk_counts = df["Risk_Level"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    risk_counts.values,
    labels=risk_counts.index,
    autopct="%1.1f%%"
)

plt.title("Risk Level Proportion")


plt.savefig(
    "risk_level_pie_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
risk_counts = df["Risk_Level"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    risk_counts.values,
    labels=risk_counts.index,
    autopct="%1.1f%%"
)

plt.title("Risk Level Proportion")

plt.savefig(
    "risk_level_pie_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
##############3
