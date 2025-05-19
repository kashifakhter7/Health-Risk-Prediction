# merged_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.markdown("""
    <style>
    .stButton>button {
        background-color: #0072C6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .st-bc {
        background-color: #e3f2fd !important;
    }
    </style>
""", unsafe_allow_html=True)


# Load trained components
model = joblib.load("model/detailed_disease_model.pkl")
transformer = joblib.load("model/detailed_scaler.pkl")
label_binarizer = joblib.load("model/disease_labels.pkl")

# Load food dataset
FOOD_DATA_PATH = "food_nutrition_dataset.csv"
food_dataset = pd.read_csv(FOOD_DATA_PATH)

# Function to get nutrition info
def get_nutrition_info(food_name):
    food_name = food_name.title()
    food_info = food_dataset[food_dataset["Food_Item"] == food_name]
    if not food_info.empty:
        return food_info
    food_info = food_dataset[food_dataset['Food_Item'].str.contains(food_name, case=False, na=False)]
    if not food_info.empty:
        return food_info
    return None

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Choose a section:", ["Health Risk Predictor", "Food Nutrition Check"])

# ----------------------------
# PAGE 1
# ----------------------------
if page == "Health Risk Predictor":
    st.markdown("<h1 style='color:#0072C6;'>🩺 Health Risk Predictor</h1>", unsafe_allow_html=True)

    with st.expander("👤 Personal Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", min_value=1)
        height = col2.number_input("Height (cm)", min_value=50, max_value=250)
        weight = col3.number_input("Weight (kg)", min_value=20, max_value=250)

        col4, col5 = st.columns(2)
        activity = col4.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        diet = col5.selectbox("Dietary Preference", ["Omnivore", "Vegetarian", "Vegan"])

    with st.expander("🍎 Nutrition Intake", expanded=True):
        col1, col2, col3 = st.columns(3)
        protein = col1.number_input("Protein (g)", min_value=0)
        sugar = col2.number_input("Sugar (g)", min_value=0)
        sodium = col3.number_input("Sodium (mg)", min_value=0)
        col4, col5 = st.columns(2)
        carbs = col4.number_input("Carbohydrates (g)", min_value=0)
        fat = col5.number_input("Fat (g)", min_value=0)

    # Calorie and BMI calculations
    calories = round((protein * 4) + (carbs * 4) + (fat * 9), 2)
    calorie_target = calories

    def calculate_bmi(height, weight):
        if height > 0:
            return weight / ((height / 100) ** 2)
        return 0

    bmi = calculate_bmi(height, weight)

    # Visual feedback
    st.success(f"🔢 **Estimated Calories Consumed**: {calories} kcal")
    st.metric("📏 BMI", f"{bmi:.1f}", help="18.5-24.9 is considered normal")

    # Interpretation
    bmi_status = "❓"
    if bmi < 18.5:
        bmi_status = "🔵 Underweight"
    elif 18.5 <= bmi <= 24.9:
        bmi_status = "🟢 Normal"
    elif 25 <= bmi <= 29.9:
        bmi_status = "🟠 Overweight"
    else:
        bmi_status = "🔴 Obese"

    st.info(f"**BMI Interpretation:** {bmi_status}")

    st.progress(min(int(calories / 50), 100), text="Daily Calorie Progress")
    st.progress(min(int(bmi / 0.4), 100), text="BMI Scale Estimate")

    colA, colB = st.columns(2)

    with colA:
        if st.button("🧠 Predict Health Risk"):
            try:
                input_df = pd.DataFrame([{
                    'Ages': age,
                    'Height': height,
                    'Weight': weight,
                    'Activity Level': activity,
                    'Dietary Preference': diet,
                    'Daily Calorie Target': calorie_target,
                    'Protein': protein,
                    'Sugar': sugar,
                    'Sodium': sodium,
                    'Carbohydrates': carbs,
                    'Fat': fat,
                }])

                input_transformed = transformer.transform(input_df)
                prediction = model.predict(input_transformed)
                predicted_labels = label_binarizer.inverse_transform(prediction)

                st.subheader("🧬 Predicted Health Risks")
                if predicted_labels[0]:
                    for label in predicted_labels[0]:
                        st.warning(f"⚠️ {label}")
                else:
                    st.success("✅ No significant health risks detected.")

                st.subheader("📊 Nutrient Summary")
                nutrients = {
                    'Calories': calories, 'Protein': protein, 'Sugar': sugar,
                    'Sodium': sodium, 'Carbohydrates': carbs, 'Fat': fat
                }
                st.write(nutrients)

                fig, ax = plt.subplots()
                ax.bar(nutrients.keys(), nutrients.values(), color='skyblue')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    with colB:
        if st.button("🥗 Suggest Meal Plan"):
            daily_calories = calorie_target
            meal_plan = {
                "🍳 Breakfast": f"{daily_calories * 0.25:.0f} kcal",
                "🍛 Lunch": f"{daily_calories * 0.35:.0f} kcal",
                "🍲 Dinner": f"{daily_calories * 0.30:.0f} kcal",
                "🍎 Snacks": f"{daily_calories * 0.10:.0f} kcal"
            }
            st.subheader("🍽️ Suggested Meal Distribution")
            st.write(meal_plan)

            # Downloadable PDF
            import pdfkit
            from tempfile import NamedTemporaryFile

            html = f"""
            <h2>Personalized Meal Plan</h2>
            <ul>
                <li><b>Breakfast</b>: {meal_plan['🍳 Breakfast']}</li>
                <li><b>Lunch</b>: {meal_plan['🍛 Lunch']}</li>
                <li><b>Dinner</b>: {meal_plan['🍲 Dinner']}</li>
                <li><b>Snacks</b>: {meal_plan['🍎 Snacks']}</li>
            </ul>
            <p><b>Total Daily Target</b>: {daily_calories:.0f} kcal</p>
            """

          # ----------------------------
# PAGE 2: FOOD NUTRITION CHECK
# ----------------------------
elif page == "Food Nutrition Check":
    st.title("🍽️ Food Nutrition Lookup")
    st.markdown("Enter the name of a food item to get its nutritional information from the dataset.")

    food_input = st.text_input("Food Name")
    if food_input:
        result = get_nutrition_info(food_input)
        if result is not None:
            st.success("✅ Found nutrition information:")
            st.dataframe(result)
        else:
            st.error("❌ Food not found in the dataset. Please try another name.")