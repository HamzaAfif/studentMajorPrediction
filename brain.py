import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from collections import Counter
from llama import generate_recommendation_explanation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson import ObjectId
import os
import seaborn as sns


client = MongoClient("mongodb://localhost:27017/")
db = client["simpleAuth"] 
students_collection = db["users"]  


def main(): 
        # Load the enhanced dataset
        df = pd.read_csv('student_data.csv')

        # Create the mapping BEFORE converting to numerical
        df['Recommended_Major'] = df['Recommended_Major'].astype('category')
        major_mapping = dict(enumerate(df['Recommended_Major'].cat.categories))
        df['Recommended_Major'] = df['Recommended_Major'].cat.codes

        # Print the correct mapping
        print("Major Mapping:")
        for key, value in major_mapping.items():
            print(f"{key}: {value}")

        # Create the mapping for preferences
        df['Preferences'] = df['Preferences'].astype('category')
        preference_mapping = dict(enumerate(df['Preferences'].cat.categories))
        df['Preferences'] = df['Preferences'].cat.codes

        '''
        # Before Conversion
        df['Recommended_Major'].value_counts().plot(kind='bar', title='Majors Before Conversion')
        plt.show()

        # After Conversion
        df['Recommended_Major'].value_counts().plot(kind='bar', title='Majors After Conversion (Numeric)')
        plt.show()

        '''

        # Define features and target
        # Include all marks and preferences as features
        X = df[['Marks_Math', 'Marks_Physics', 'Marks_Chemistry', 'Marks_Art', 'Marks_Economics', 'Preferences']]
        y = df['Recommended_Major']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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

        # Check class distribution in training data
        print("Class distribution in training data:", Counter(y_train))

        # Train the final model
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = knn.predict(X_test_scaled)
        print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Predict for a new student
        # Adjust for the new feature set
        new_student = pd.DataFrame([[80, 75, 85, 60, 78, 1]], 
                                columns=["Marks_Math", "Marks_Physics", "Marks_Chemistry", "Marks_Art", "Marks_Economics", "Preferences"])

        # Scale the input
        new_student_scaled = scaler.transform(new_student)

        # Predict the major
        predicted_major = knn.predict(new_student_scaled)

        # Debug: Inspect nearest neighbors
        distances, indices = knn.kneighbors(new_student_scaled)
        neighbor_classes = y_train.iloc[indices[0]]

        print("Distances to nearest neighbors:", distances)
        print("Indices of nearest neighbors:", indices)
        print("Classes of nearest neighbors:", neighbor_classes)

        # Map the predicted number to the corresponding major name
        predicted_major_name = major_mapping.get(predicted_major[0], "Unknown Major")
        print(f"\nPredicted Major for new student: {predicted_major_name}")

        # Optional: Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=major_mapping.values())
        disp.plot()



        neighbor_major_counts = neighbor_classes.value_counts()

        neighbor_summary = ", ".join(
            [f"{count} students in '{major_mapping[major]}'" for major, count in neighbor_major_counts.items()]
        )

        print(generate_recommendation_explanation(
            new_student=new_student,
            predicted_major_name=predicted_major_name,
            neighbor_summary=neighbor_summary
        ))


        print()
        print()

        kmeans = KMeans(n_clusters=5, random_state=42)  # 5 clusters for 5 majors
        clusters = kmeans.fit_predict(X)

        # Add cluster labels to the dataset
        df['Cluster'] = clusters

        print("\nK-Means Clustering Results:")
        print(df[['Marks_Math', 'Marks_Physics', 'Marks_Chemistry', 'Marks_Art', 'Marks_Economics', 'Preferences', 'Cluster']])

        # Analyze cluster composition
        cluster_summary = df.groupby('Cluster')['Recommended_Major'].value_counts()
        print("\nCluster Composition by Recommended Major:")
        print(cluster_summary)


        # Reduce dimensionality with PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Visualize the clusters
        plt.figure(figsize=(8, 6))
        for cluster_id in range(kmeans.n_clusters):
            plt.scatter(
                X_pca[clusters == cluster_id, 0],
                X_pca[clusters == cluster_id, 1],
                label=f"Cluster {cluster_id}"
            )

        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            color='red',
            marker='X',
            s=200,
            label='Centroids'
        )
        plt.title("Student Clusters Based on Academic Profiles")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()

        # Describe clusters
        cluster_profiles = df.groupby('Cluster').mean()
        print("\nCluster Profiles (Average Marks and Preferences):")
        print(cluster_profiles)


        # Predict the cluster for the new student
        new_student_cluster = kmeans.predict(new_student)[0]
        print(f"\nNew Student belongs to Cluster {new_student_cluster}")

        
        print(f"\nCluster {new_student_cluster} Profile (Average Marks and Preferences):")
        print(cluster_profiles.loc[new_student_cluster])

        '''
        retrieved_neighbors = y_train.iloc[indices[0]]

        # Step 2: Aggregate Neighbor Insights
        neighbor_major_counts = retrieved_neighbors.value_counts()
        recommended_major = neighbor_major_counts.idxmax()

        # Step 3: Generate Explanation
        def generate_explanation(student_input, recommended_major, neighbor_major_counts):
            explanation = (
                f"Based on your input marks: {student_input.values.flatten()} and preferences, "
                f"the recommended major is '{major_mapping[recommended_major]}'. "
                f"Out of your nearest neighbors, {neighbor_major_counts[recommended_major]} students "
                f"also pursued '{major_mapping[recommended_major]}', indicating a strong match."
            )
            return explanation

        # Generate Explanation
        new_student_input = new_student.iloc[0]
        explanation = generate_explanation(new_student_input, recommended_major, neighbor_major_counts)

        print("Explanation:", explanation)

        '''

def predict_student_major(student_data, student_id):
    
    html_logs = []

    
    df = pd.read_csv('student_data.csv')

    df['Recommended_Major'] = df['Recommended_Major'].astype('category')
    major_mapping = dict(enumerate(df['Recommended_Major'].cat.categories))
    df['Recommended_Major'] = df['Recommended_Major'].cat.codes

    html_logs.append('<h3 style="color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px;">Major Mapping:</h3>')
    for key, value in major_mapping.items():
        html_logs.append(f'<p style="margin: 5px 0; font-size: 14px; color: #34495e;">{key}: {value}</p>')

    df['Preferences'] = df['Preferences'].astype('category')
    preference_mapping = dict(enumerate(df['Preferences'].cat.categories))
    df['Preferences'] = df['Preferences'].cat.codes
    
    plot_dir = os.path.join('web', 'public', 'plots', str(student_id))
    os.makedirs(plot_dir, exist_ok=True)
    
    major_plot_path = os.path.join(plot_dir, 'recommended_major_distribution.png')
    plt.figure()
    df['Recommended_Major'].replace(major_mapping).value_counts().plot(
        kind='bar', title='Recommended Major Distribution'
    )
    plt.xlabel('Majors')
    plt.ylabel('Count')
    plt.savefig(major_plot_path)
    plt.close()

    preferences_plot_path = os.path.join(plot_dir, 'preferences_distribution.png')
    plt.figure()
    df['Preferences'].replace(preference_mapping).value_counts().plot(
        kind='bar', title='Preferences Distribution'
    )
    plt.xlabel('Preferences')
    plt.ylabel('Count')
    plt.savefig(preferences_plot_path)
    plt.close()
    
    

    X = df[['Marks_Math', 'Marks_Physics', 'Marks_Chemistry', 'Marks_Art', 'Marks_Economics', 'Preferences']]
    y = df['Recommended_Major']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    sizes = [0.7, 0.3]  
    labels = ['Training Data', 'Testing Data']
    train_test_plot_path = os.path.join(plot_dir, 'train_test_split.png') 
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Train-Test Split')
    plt.savefig(train_test_plot_path)  
    plt.close()  

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    before_scaling_plot_path = os.path.join(plot_dir, 'before_scaling.png')  
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=X_train, orient='h')
    plt.title("Before Scaling")
    plt.savefig(before_scaling_plot_path)  
    plt.close()  

    after_scaling_plot_path = os.path.join(plot_dir, 'after_scaling.png')  
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=pd.DataFrame(X_train_scaled, columns=X_train.columns), orient='h')
    plt.title("After Scaling")
    plt.savefig(after_scaling_plot_path)  
    plt.close()  

    k_values = list(range(1, 20))
    f1_scores = []
    best_k = 1
    best_score = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
        mean_score = scores.mean()
        f1_scores.append(mean_score)  # Collect scores for plotting

        if mean_score > best_score:  # Update best_k and best_score
            best_k = k
            best_score = mean_score

    html_logs.append(f'<h3 style="color: #27ae60; margin-top: 15px;">Best k:</h3>')
    html_logs.append(f'<p style="font-size: 14px; color: #34495e;">{best_k}, Best Cross-Validation F1-Score: {best_score:.2f}</p>')

    best_k_plot_path = os.path.join(plot_dir, 'best_k_for_knn.png')
    plt.plot(k_values, f1_scores, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean F1-Score')
    plt.title('Finding the Best k for KNN')
    plt.grid(True)
    plt.savefig(best_k_plot_path)
    plt.close()

    
    
    # Save Class Distribution in Training Data
    class_distribution_plot_path = os.path.join(plot_dir, 'class_distribution_training_data.png') 
    class_counts = Counter(y_train)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training Data')
    plt.savefig(class_distribution_plot_path) 
    plt.close() 


    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    
    
    pca_plot_path = os.path.join(plot_dir, 'pca_training_data.png')  
    pca = PCA(n_components=2)
    X_train_2D = pca.fit_transform(X_train_scaled)

   
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='viridis', s=50, alpha=0.8)
    plt.title("Training Data Points (Reduced to 2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Recommended Major")
    plt.savefig(pca_plot_path) 
    plt.close()  

    y_pred = knn.predict(X_test_scaled)
    html_logs.append(f'<h3 style="color: #2980b9; margin-top: 15px;">Final Model Accuracy:</h3>')
    html_logs.append(f'<p style="font-size: 14px; color: #34495e;">{accuracy_score(y_test, y_pred):.2f}</p>')
    html_logs.append('<h3 style="color: #8e44ad; margin-top: 15px;">Classification Report:</h3>')
    html_logs.append(f'<pre style="background-color: #ecf0f1; padding: 10px; border-radius: 5px; font-size: 12px;">{classification_report(y_test, y_pred, zero_division=0)}</pre>')
    
    confusion_matrix_plot_path = os.path.join(plot_dir, 'confusion_matrix.png')  
    cm_display = ConfusionMatrixDisplay.from_estimator(knn, X_test_scaled, y_test, display_labels=major_mapping.values())
    cm_display.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_plot_path)  
    plt.close()  

    new_student = pd.DataFrame([student_data], 
                               columns=["Marks_Math", "Marks_Physics", "Marks_Chemistry", "Marks_Art", "Marks_Economics", "Preferences"])

    new_student_scaled = scaler.transform(new_student)

    predicted_major = knn.predict(new_student_scaled)

    distances, indices = knn.kneighbors(new_student_scaled)
    neighbor_classes = y_train.iloc[indices[0]]
    
    neighbor_summary = f"Distances: {distances.tolist()}, Classes: {neighbor_classes.to_list()}"

    
    
    html_logs.append('<h3 style="color: #c0392b; margin-top: 15px;">Nearest Neighbors:</h3>')
    html_logs.append(f'<p style="font-size: 14px; color: #34495e;">Distances to nearest neighbors: {distances}</p>')
    html_logs.append(f'<p style="font-size: 14px; color: #34495e;">Indices of nearest neighbors: {indices}</p>')
    html_logs.append(f'<p style="font-size: 14px; color: #34495e;">Classes of nearest neighbors: {neighbor_classes.to_list()}</p>')

    predicted_major_name = major_mapping.get(predicted_major[0], "Unknown Major")
    html_logs.append(f'<h3 style="color: #d35400; margin-top: 15px;">Predicted Major for new student:</h3>')
    html_logs.append(f'<p style="font-size: 16px; font-weight: bold; color: #34495e;">{predicted_major_name}</p>')
    
    llm_reply = generate_recommendation_explanation(student_data, predicted_major_name, neighbor_summary)
    
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    
    clusters = kmeans.fit_predict(X)
    
    df['Cluster'] = clusters
    
    cluster_summary = df.groupby('Cluster')['Recommended_Major'].value_counts()
    html_logs.append(f"<h3 style='color: #2980b9;'>Cluster Summary:</h3>")
    html_logs.append(f"<pre>{cluster_summary}</pre>")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plot_dir = os.path.join('web', 'public', 'plots', str(student_id))
    os.makedirs(plot_dir, exist_ok=True)
    
    cluster_plot_path = os.path.join(plot_dir, 'student_clusters.png')
    plt.figure(figsize=(8, 6))
    for cluster_id in range(kmeans.n_clusters):
        plt.scatter(
            X_pca[clusters == cluster_id, 0],
            X_pca[clusters == cluster_id, 1],
            label=f"Cluster {cluster_id}"
        )
        
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        color='red',
        marker='X',
        s=200,
        label='Centroids'
    )
    plt.title("Student Clusters Based on Academic Profiles")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(cluster_plot_path)
    plt.close()
    
    cluster_plot_path = cluster_plot_path.replace('web\\', '')

    html_logs.append(f"<h3 style='color: #2980b9;'>Cluster Plot:</h3>")
    html_logs.append(f"<img src='/{cluster_plot_path}' alt='Student Clusters' style='width: 100%; max-width: 600px;' />")

    new_student = pd.DataFrame([student_data], 
                               columns=["Marks_Math", "Marks_Physics", "Marks_Chemistry", "Marks_Art", "Marks_Economics", "Preferences"])
    new_student_cluster = kmeans.predict(new_student)[0]
    
    html_logs.append(f"<h3 style='color: #c0392b;'>New Student's Predicted Cluster:</h3>")
    html_logs.append(f"<p style='font-size: 14px; color: #34495e;'>New student belongs to Cluster {new_student_cluster}.</p>")

    html_content = """
    <div style="font-family: Arial, sans-serif; line-height: 1.6; padding: 15px; background-color: #f4f6f7; border-radius: 10px; max-width: 800px; margin: 0 auto; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
    """ + "".join(html_logs) + "</div>"
    
    plot_paths = {
        "recommended_major_distribution": major_plot_path,
        "preferences_distribution": preferences_plot_path,
        "train_test_split": train_test_plot_path,
        "before_scaling": before_scaling_plot_path,
        "after_scaling" : after_scaling_plot_path,
        "best_k_for_knn" : best_k_plot_path,
        "class_distribution_training_data" : class_distribution_plot_path,
        "pca_training_data": pca_plot_path,
        "confusion_matrix": confusion_matrix_plot_path,
        "student_clusters": cluster_plot_path
    }
    
    try:
        result = students_collection.update_one(
            {"_id": ObjectId(student_id)},
            {"$set": {"prediction_report": html_content, "llmreply": llm_reply, "plots": plot_paths}}
        )

        if result.matched_count == 0:
            print(f"Student with ID {student_id} not found.")
        else:
            print(f"Prediction report and LLM reply saved for student ID {student_id}.")
    except Exception as e:
        print(f"Error saving prediction report: {e}")
        

    return html_content



if __name__ == "__main__":
    main()