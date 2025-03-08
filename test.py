import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from llama import generate_recommendation_explanation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load the enhanced dataset
df = pd.read_csv('student_data.csv')

# Create the mapping BEFORE converting to numerical
df['Recommended_Major'] = df['Recommended_Major'].astype('category')
major_mapping = dict(enumerate(df['Recommended_Major'].cat.categories))
df['Recommended_Major'] = df['Recommended_Major'].cat.codes


print("Major Mapping:")
for key, value in major_mapping.items():
    print(f"{key}: {value}")

# Create the mapping for preferences
df['Preferences'] = df['Preferences'].astype('category')
preference_mapping = dict(enumerate(df['Preferences'].cat.categories))
df['Preferences'] = df['Preferences'].cat.codes


# This graph shows how many students are in each "Recommended Major" 
plt.figure()
df['Recommended_Major'].replace(major_mapping).value_counts().plot(
    kind='bar', title='Recommended Major Distribution'
)
plt.xlabel('Majors')
plt.ylabel('Count')
plt.show()

# This graph shows how many students chose each "Preference"
plt.figure()
df['Preferences'].replace(preference_mapping).value_counts().plot(
    kind='bar', title='Preferences Distribution'
)
plt.xlabel('Preferences')
plt.ylabel('Count')
plt.show()


# Define features and target
# Include all marks and preferences as features
X = df[['Marks_Math', 'Marks_Physics', 'Marks_Chemistry', 'Marks_Art', 'Marks_Economics', 'Preferences']]
y = df['Recommended_Major']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualize the train-test split
sizes = [0.7, 0.3]  # Training = 70%, Testing = 30%
labels = ['Training Data', 'Testing Data']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Train-Test Split')
plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import seaborn as sns

# Before scaling
plt.figure(figsize=(10, 5))
sns.boxplot(data=X_train, orient='h')
plt.title("Before Scaling")
plt.show()

# After scaling
plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(X_train_scaled, columns=X_train.columns), orient='h')
plt.title("After Scaling")
plt.show()


# Hyperparameter tuning
best_k = 1
best_score = 0
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    mean_score = scores.mean()
    if mean_score > best_score:
        best_k = k
        best_score = mean_score

print(f"Best k: {best_k}, Best Cross-Validation F1-Score: {best_score:.2f}")

k_values = list(range(1, 20))
f1_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    f1_scores.append(scores.mean())

plt.plot(k_values, f1_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean F1-Score')
plt.title('Finding the Best k for KNN')
plt.grid(True)
plt.show()


# Check class distribution in training data
print("Class distribution in training data:", Counter(y_train))


class_counts = Counter(y_train)
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution in Training Data')
plt.show()


# Train the final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)


from sklearn.decomposition import PCA

# Reduce the data to 2D for visualization
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train_scaled)

# Scatter plot of training data
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='viridis', s=50)
plt.title("Training Data Points (Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Recommended Major")
plt.show()


# Evaluate the model
y_pred = knn.predict(X_test_scaled)
print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

from sklearn.metrics import ConfusionMatrixDisplay

# Generate and display confusion matrix
cm_display = ConfusionMatrixDisplay.from_estimator(knn, X_test_scaled, y_test, display_labels=major_mapping.values())
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
