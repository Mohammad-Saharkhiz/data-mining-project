# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('modified_diabetes_prediction_dataset.csv')

# Data Preprocessing
# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Replace 'unknown' with NaN
df.replace('unknown', np.nan, inplace=True)

# Handle missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'unknown': -1})
df['smoking_history'] = df['smoking_history'].map({'never': 0, 'No Info': -1, 'current': 1, 'former': 2, 'not current': 3, 'ever': 4})

# Check for missing values after handling
print("Missing values after handling:")
print(df.isnull().sum())

# حذف سطرهای حاوی NaN از df
df.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('diabetes', axis=1))

# Convert scaled data to numpy array
df_scaled_np = np.array(df_scaled)

# بررسی طول df و df_scaled_np
print("Length of df:", len(df))
print("Length of df_scaled_np:", len(df_scaled_np))

# Clustering with K-means
# Determine the optimal number of clusters using the Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # تنظیم n_init برای جلوگیری از اخطار
    kmeans.fit(df_scaled_np)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, 11), y=inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # تنظیم n_init برای جلوگیری از اخطار
df['kmeans_cluster'] = kmeans.fit_predict(df_scaled_np)

# Analyze the clusters using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_scaled_np[:, 0], y=df_scaled_np[:, 1], hue=df['kmeans_cluster'], palette='viridis')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Hierarchical Clustering
# نمونه‌گیری تصادفی از داده‌ها (1% of the data)
sample_size = int(0.001 * len(df_scaled_np))  # 1% of the data
random_indices = np.random.choice(len(df_scaled_np), sample_size, replace=False)
df_scaled_sample = df_scaled_np[random_indices]

# Perform hierarchical clustering
linked = linkage(df_scaled_sample, method='ward')

# Plot the dendrogram using seaborn
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Classification
# Split the data into training and testing sets
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert training and testing data to numpy arrays
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_np, y_train_np)
y_pred_dt = dt_classifier.predict(X_test_np)

# Evaluate Decision Tree Classifier
print("Decision Tree Classifier:")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_dt)}")
print(f"Precision: {precision_score(y_test_np, y_pred_dt)}")
print(f"Recall: {recall_score(y_test_np, y_pred_dt)}")
print(f"F1-Score: {f1_score(y_test_np, y_pred_dt)}")

# Plot Confusion Matrix for Decision Tree using seaborn
conf_matrix_dt = confusion_matrix(y_test_np, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_np, y_train_np)
y_pred_svm = svm_classifier.predict(X_test_np)

# Evaluate SVM Classifier
print("SVM Classifier:")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_svm)}")
print(f"Precision: {precision_score(y_test_np, y_pred_svm)}")
print(f"Recall: {recall_score(y_test_np, y_pred_svm)}")
print(f"F1-Score: {f1_score(y_test_np, y_pred_svm)}")

# Plot Confusion Matrix for SVM using seaborn
conf_matrix_svm = confusion_matrix(y_test_np, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # تنظیم تعداد همسایگان (K)
knn_classifier.fit(X_train_np, y_train_np)
y_pred_knn = knn_classifier.predict(X_test_np)

# Evaluate KNN Classifier
print("KNN Classifier:")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_knn)}")
print(f"Precision: {precision_score(y_test_np, y_pred_knn)}")
print(f"Recall: {recall_score(y_test_np, y_pred_knn)}")
print(f"F1-Score: {f1_score(y_test_np, y_pred_knn)}")

# Plot Confusion Matrix for KNN using seaborn
conf_matrix_knn = confusion_matrix(y_test_np, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Compare the results
results = {
    'Model': ['Decision Tree', 'SVM', 'KNN'],
    'Accuracy': [accuracy_score(y_test_np, y_pred_dt), accuracy_score(y_test_np, y_pred_svm), accuracy_score(y_test_np, y_pred_knn)],
    'Precision': [precision_score(y_test_np, y_pred_dt), precision_score(y_test_np, y_pred_svm), precision_score(y_test_np, y_pred_knn)],
    'Recall': [recall_score(y_test_np, y_pred_dt), recall_score(y_test_np, y_pred_svm), recall_score(y_test_np, y_pred_knn)],
    'F1-Score': [f1_score(y_test_np, y_pred_dt), f1_score(y_test_np, y_pred_svm), f1_score(y_test_np, y_pred_knn)]
}

results_df = pd.DataFrame(results)
print(results_df)
