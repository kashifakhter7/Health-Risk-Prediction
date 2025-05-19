# merged_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained components
model = joblib.load("detailed_disease_model.pkl")
transformer = joblib.load("detailed_scaler.pkl")
label_binarizer = joblib.load("disease_labels.pkl")

# Load food dataset
FOOD_DATA_PATH = "food_nutrition_dataset.csv"
food_dataset = pd.read_csv(FOOD_DATA_PATH)

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
    calorie_target = st.number_input("Daily Calorie Target", min_value=1000, max_value=5000)

    protein = st.number_input("Protein (g)", min_value=0)
    sugar = st.number_input("Sugar (g)", min_value=0)
    sodium = st.number_input("Sodium (mg)", min_value=0)
    carbs = st.number_input("Carbohydrates (g)", min_value=0)
    fat = st.number_input("Fat (g)", min_value=0)
    fiber = st.number_input("Fiber (g)", min_value=0)

    # Calculate calories dynamically
    calories = round((protein * 4) + (carbs * 4) + (fat * 9) + (fiber * 2), 2)
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
                'Calories': calories,
                'Protein': protein,
                'Sugar': sugar,
                'Sodium': sodium,
                'Carbohydrates': carbs,
                'Fat': fat,
                'Fiber': fiber
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
                'Sodium': sodium, 'Carbohydrates': carbs, 'Fat': fat, 'Fiber': fiber
            }
            st.write(nutrients)

            fig, ax = plt.subplots()
            ax.bar(nutrients.keys(), nutrients.values(), color='lightblue')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ----------------------------
# PAGE 2: FOOD NUTRITION LOOKUP
# ----------------------------
elif page == "Food Nutrition Lookup":
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
