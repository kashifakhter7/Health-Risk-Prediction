import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model/detailed_disease_model.pkl")
transformer = joblib.load("model/detailed_scaler.pkl")
label_binarizer = joblib.load("model/disease_labels.pkl")
food_data = pd.read_csv("food_nutrition_dataset.csv")

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Health Risk Predictor", "Food Nutrition Checker"])

# Utility functions
def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2) if height > 0 else 0

def get_nutrition_info(food_name):
    food_name = food_name.title()
    match = food_data[food_data["Food_Item"].str.contains(food_name, case=False, na=False)]
    return match if not match.empty else None

# -------------- Health Risk Page --------------
if page == "Health Risk Predictor":
    st.title("🩺 Health Risk Predictor")

    col1, col2 = st.columns(2)

    # --- Personal Info ---
    with col1:
        st.subheader("👤 Personal Info")
        age = st.number_input("Age", 1)
        height = st.number_input("Height (cm)", 50, 250)
        weight = st.number_input("Weight (kg)", 20, 250)
        activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        diet = st.selectbox("Dietary Preference", ["Omnivore", "Vegetarian", "Vegan"])

        bmi = calculate_bmi(height, weight)
        st.metric("BMI", f"{bmi:.1f}")

    # --- Nutrition Info ---
    with col2:
        st.subheader("🍎 Nutrition Intake")
        protein = st.number_input("Protein (g)", 0)
        sugar = st.number_input("Sugar (g)", 0)
        sodium = st.number_input("Sodium (mg)", 0)
        carbs = st.number_input("Carbs (g)", 0)
        fat = st.number_input("Fat (g)", 0)

        calories = round(protein * 4 + carbs * 4 + fat * 9, 2)
        st.success(f"Estimated Calories: {calories} kcal")

    # --- Prediction ---
    if st.button("Predict Health Risk"):
        input_df = pd.DataFrame([{
            'Ages': age,
            'Height': height,
            'Weight': weight,
            'Activity Level': activity,
            'Dietary Preference': diet,
            'Daily Calorie Target': calories,
            'Protein': protein,
            'Sugar': sugar,
            'Sodium': sodium,
            'Carbohydrates': carbs,
            'Fat': fat,
        }])

        try:
            X_input = transformer.transform(input_df)
            y_pred = model.predict(X_input)
            results = label_binarizer.inverse_transform(y_pred)

            st.subheader("🔬 Predicted Health Risks")
            if results[0]:
                for r in results[0]:
                    st.warning(r)
            else:
                st.success("No significant risks detected.")

            # Bar chart
            st.subheader("Nutrition Chart")
            nutrients = {
                'Calories': calories,
                'Protein': protein,
                'Sugar': sugar,
                'Sodium': sodium,
                'Carbohydrates': carbs,
                'Fat': fat
            }
            fig, ax = plt.subplots()
            ax.bar(nutrients.keys(), nutrients.values(), color='teal')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # --- Meal Plan Suggestion ---
    if st.button("Suggest Meal Plan"):
        meal_plan = {
            "🍳 Breakfast": f"{calories * 0.25:.0f} kcal",
            "🍛 Lunch": f"{calories * 0.35:.0f} kcal",
            "🍲 Dinner": f"{calories * 0.30:.0f} kcal",
            "🍎 Snacks": f"{calories * 0.10:.0f} kcal"
        }

        st.subheader("🍽️ Suggested Meal Plan")
        for meal, value in meal_plan.items():
            st.write(f"{meal}: {value}")

        # Optional pie chart
        st.subheader("📊 Calorie Distribution")
        fig, ax = plt.subplots()
        ax.pie([0.25, 0.35, 0.30, 0.10],
               labels=meal_plan.keys(),
               autopct='%1.0f%%',
               colors=['#ffd54f', '#4fc3f7', '#81c784', '#ff8a65'])
        ax.axis('equal')
        st.pyplot(fig)

# -------------- Nutrition Info Page --------------
elif page == "Food Nutrition Checker":
    st.title("🥗 Food Nutrition Checker")

    food_name = st.text_input("Enter a food item:")
    if food_name:
        info = get_nutrition_info(food_name)
        if info is not None:
            st.success("Nutritional info found:")
            st.dataframe(info)
        else:
            st.error("Food not found in dataset.")
