# merged_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# PAGE 1: HEALTH RISK PREDICTOR
# ----------------------------
if page == "Health Risk Predictor":
    st.title("Health Risk Predictor And Food Nutrition Checker")
    st.markdown("Enter your nutritional and demographic data below to predict potential health risks.")

    # Input fields
    age = st.number_input("Age", min_value=1)
    height = st.number_input("Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=250)
    activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    diet = st.selectbox("Dietary Preference", ["Omnivore", "Vegetarian", "Vegan"])
    #calorie_target = st.number_input("Daily Calorie Target", min_value=1000, max_value=5000)

    protein = st.number_input("Protein (g)", min_value=0)
    sugar = st.number_input("Sugar (g)", min_value=0)
    sodium = st.number_input("Sodium (mg)", min_value=0)
    carbs = st.number_input("Carbohydrates (g)", min_value=0)
    fat = st.number_input("Fat (g)", min_value=0)
    # Calculate calories dynamically
    calories = round((protein * 4) + (carbs * 4) + (fat * 9) , 2)
    st.markdown(f"### 🔢 Estimated Calories Consumed: **{calories} kcal**")
    calorie_target = calories

        # Calculate BMI and show
    def calculate_bmi(height, weight):
        if height > 0:
            return weight / ((height / 100) ** 2)
        return 0

    bmi = calculate_bmi(height, weight)
    st.metric("BMI", f"{bmi:.1f}", help="18.5-24.9 is considered normal")

    # Suggest Meal Plan Button
    if st.button("Suggest Meal Plan"):
        daily_calories = calorie_target
        meal_plan = {
            "Breakfast": f"{daily_calories * 0.25:.0f} kcal",
            "Lunch": f"{daily_calories * 0.35:.0f} kcal",
            "Dinner": f"{daily_calories * 0.30:.0f} kcal",
            "Snacks": f"{daily_calories * 0.10:.0f} kcal"
        }
        st.subheader("Suggested Meal Distribution")
        st.write(meal_plan)



    # Calculate calories dynamically
    calories = round((protein * 4) + (carbs * 4) + (fat * 9) , 2)
    st.markdown(f"### 🔢 Estimated Calories Consumed: **{calories} kcal**")


    # Predict button
    if st.button("Predict Health Risk"):
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

            st.subheader("Predicted Health Risks")
            if predicted_labels[0]:
                for label in predicted_labels[0]:
                    st.warning(f"⚠️ {label}")
            else:
                st.success("✅ No significant health risks detected.")

            st.subheader("Nutrient Summary")
            nutrients = {
                'Calories': calories, 'Protein': protein, 'Sugar': sugar,
                'Sodium': sodium, 'Carbohydrates': carbs, 'Fat': fat
            }
            st.write(nutrients)

            fig, ax = plt.subplots()
            ax.bar(nutrients.keys(), nutrients.values(), color='lightblue')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

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



