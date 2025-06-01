import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

# Define models
models = {
    'Support Vector Machine': MultiOutputClassifier(SVC()),
    'Random Forest': MultiOutputClassifier(RandomForestClassifier()),
    'Logistic Regression': MultiOutputClassifier(LogisticRegression()),
    'K-Nearest Neighbors': MultiOutputClassifier(KNeighborsClassifier()),
    'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier()),
    'Naive Bayes': MultiOutputClassifier(GaussianNB()),
    'AdaBoost': MultiOutputClassifier(AdaBoostClassifier()),
    'Gradient Boosting': MultiOutputClassifier(GradientBoostingClassifier()),
    'Extra Trees': MultiOutputClassifier(ExtraTreesClassifier()),
    'Voting Classifier': MultiOutputClassifier(VotingClassifier(estimators=[('svm', SVC()), ('rf', RandomForestClassifier()), ('lr', LogisticRegression())])),
    'XGBoost': MultiOutputClassifier(XGBClassifier())
}

food= pd.read_csv('food.csv')
d= pd.read_csv('d.csv')

food.shape

d.shape

d.head()

# Drop the specified columns
d = d.drop(columns=['Breakfast Suggestion','Calories' ,'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion'])

d.head()

d['Fiber'].mean()

d['Fiber'].median()

# Process the target column (Disease)
d['Disease'] = d['Disease'].fillna('None')
d['Disease'] = d['Disease'].apply(lambda x: [d.strip() for d in x.split(',')])
print("✅ Disease column cleaned and split into lists.")

# Select relevant features for training
features = [
    'Ages', 'Height', 'Weight', 'Activity Level', 'Dietary Preference', 'Daily Calorie Target','Protein', 'Sugar', 'Sodium',
    'Carbohydrates', 'Fat']
X = d[features]
print("Feature columns selected.")

X.head()

#  Encode target labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(d['Disease'])
os.makedirs("model", exist_ok=True)
joblib.dump(mlb, "model/disease_labels.pkl")
print("Disease labels encoded and saved.")

# Import required preprocessing tools
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# define categorical and numerical columns
categorical_cols = ['Activity Level', 'Dietary Preference']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

#Create the ColumnTransformer
column_transformer = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
])

#Fit and transform the features
X_prepared = column_transformer.fit_transform(X)

#Save the transformer for future use
joblib.dump(column_transformer, "model/detailed_scaler.pkl")
print("Features scaled and encoded, transformer saved.")

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)
print("Data split into training and test sets.")

# Train and evaluate models
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)


# Find the best model
best_model = max(accuracies, key=accuracies.get)

# Print accuracy of each model and the best model
print("Model Accuracies:")
for model, acc in accuracies.items():
    print(f"{model}: {acc:.4f}")
    print("Precision score:")


print(f"\nBest Model: {best_model} with accuracy {accuracies[best_model]:.4f}")

from sklearn.metrics import accuracy_score, precision_score

accuracies = {}
precision_scores_value = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    precision_scores_value[name] = precision_score(y_test, y_pred, average='macro')  # or 'weighted'

# Find the best model
best_model = max(accuracies, key=accuracies.get)

# Print accuracy and precision for each model
print("Model Accuracies and Precision Scores:")
for model in models:
    print(f"{model} - Accuracy: {accuracies[model]:.4f}, Precision: {precision_scores_value[model]:.4f}")

print(f"\nBest Model: {best_model} with accuracy {accuracies[best_model]:.4f}")

model= DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = DT.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "model/detailed_disease_model.pkl")
print("Trained model saved to 'model/detailed_disease_model.pkl'.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Store metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': []
}

# To store confusion matrices
confusion_matrices = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # Save metrics and confusion matrix
    metrics['Model'].append(name)
    metrics['Accuracy'].append(acc)
    metrics['Precision'].append(prec)
    metrics['Recall'].append(rec)
    confusion_matrices[name] = cm

# Convert to DataFrame
results_df = pd.DataFrame(metrics)
print(results_df)

# Plotting accuracy, precision, recall
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.barplot(data=results_df, y='Model', x='Accuracy', ax=axes[0], color='skyblue')
axes[0].set_title("Model Accuracy")

sns.barplot(data=results_df, y='Model', x='Precision', ax=axes[1], color='salmon')
axes[1].set_title("Model Precision")

sns.barplot(data=results_df, y='Model', x='Recall', ax=axes[2], color='lightgreen')
axes[2].set_title("Model Recall")

plt.tight_layout()
plt.show()

# Plot confusion matrices for all models
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()