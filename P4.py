import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Load data from CSV file using pandas
data = pd.read_csv("tennisdata.csv")

# Separate features and target variable
X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)

# Get feature names after one-hot encoding
feature_names = X_encoded.columns
print(feature_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Print the decision tree
tree_text = export_text(classifier, feature_names=feature_names)
print("Decision Tree:\n", tree_text)

# Predict the target variable for new instances
new_instance = {"Outlook_Sunny": 1, "Temperature_Hot": 1, "Humidity_High": 1, "Windy_Weak": 1}
new_instance_encoded = pd.DataFrame([new_instance], columns=feature_names)
new_instance_prediction = classifier.predict(new_instance_encoded)

print("Classification:", new_instance_prediction[0])
