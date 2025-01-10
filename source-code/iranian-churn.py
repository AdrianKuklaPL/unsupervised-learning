# Necessary imports
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)         # For NumPy operations
random.seed(seed)            # For Python's built-in random operations

# Fetch dataset - Iranian churn
iranian_churn = fetch_ucirepo(id=563) 

# Data (as pandas dataframes)
X = iranian_churn.data.features
y = iranian_churn.data.targets

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
columns_X = ["Call Failure", "Complains", "Subscription Length", "Charge Amount", "Seconds of Use",
            "Frequency of Use", "Frequency of SMS", "Distinct Called Numbers", "Age Group", "Tariff Plan",
            "Status", "Age", "Customer Value"]
X_scaled = pd.DataFrame(X_scaled, columns=columns_X)
Y = y.values.ravel()

############## K-Means Clustering ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set up a range of cluster numbers to evaluate
range_n_clusters = list(range(2, 10))
inertia_values = []
silhouette_scores = []

# Evaluate KMeans clustering for different numbers of clusters on the training set
for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)
    
    # Calculate the average silhouette score on the training set
    silhouette_avg = silhouette_score(X_train, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Plotting the Elbow Method graph (inertia vs. number of clusters)
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Training Set)')

# Adding labels to the points
for i, k in enumerate(range_n_clusters):
    plt.text(k, inertia_values[i], str(k), fontsize=9, verticalalignment='bottom')

plt.grid(True)
plt.show()

# Plotting the Silhouette Score graph (silhouette score vs. number of clusters)
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k (Training Set)')

# Adding labels to the points
for i, k in enumerate(range_n_clusters):
    plt.text(k, silhouette_scores[i], str(k), fontsize=9, verticalalignment='bottom')

plt.grid(True)
plt.show()

# Fit the K-Means model with the chosen number of clusters on the training set (e.g., k = 5)
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Predict the clusters on the test set
Y_pred = kmeans.predict(X_test)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0403

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1569

########### Expectation Maximization ###########
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define range of potential clusters
range_n_components = range(2, 11)

# Lists to store the BIC scores and Silhouette scores for each number of components
bic_scores = []
silhouette_scores = []

# Loop over the range of components
for n in range_n_components:
    # Initialize and fit the GMM on the training data
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=seed)
    gmm.fit(X_train)
    
    # Calculate BIC score and append it to the list (use training data)
    bic = gmm.bic(X_train)
    bic_scores.append(bic)
    
    # Predict cluster assignments on training data
    cluster_labels = gmm.predict(X_train)
    
    # Calculate silhouette score and append it to the list (use training data)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting BIC scores (Elbow Method)
plt.figure(figsize=(10, 5))
plt.plot(range_n_components, bic_scores, marker='o')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.title('BIC for GMM (Training Set)')
plt.xticks(range_n_components)
plt.grid(True)
plt.show()

# Plotting Silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range_n_components, silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for GMM (Training Set)')
plt.xticks(range_n_components)
plt.grid(True)
plt.show()

# Fit the GMM model with the optimal number of components on the training data
optimal_component = 8
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Predict cluster assignments on the test set
Y_pred = gmm.predict(X_test)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.1077

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") #0.1162

# Assuming you have a dataset 'X' and you've fitted the GMM model:
gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(X)

# Get the predicted cluster labels for each instance
labels = gmm.predict(X)

# Count the number of instances in each cluster
cluster_counts = np.bincount(labels)

# Display the counts per cluster
for cluster_id, count in enumerate(cluster_counts):
    print(f"Cluster {cluster_id}: {count} instances")


############ PCA ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply PCA on the training data
pca = PCA()
X_train_pca = pca.fit_transform(X_train)

# Transform the test data using the fitted PCA
X_test_pca = pca.transform(X_test)

# Explained variance ratio for each principal component (on the training data)
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
# Plotting explained variance ratio as a bar chart with cumulative variance labels
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
# Add cumulative variance labels on top of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{cumulative_variance[i]:.2f}', 
             ha='center', va='bottom', fontsize=10)

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component (Training Set)')
plt.show()

# 5 number of components are enough to explain 80% of the variance

##################### ICA #################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply ICA to the training data
ica = FastICA(random_state=seed)
X_train_ica = ica.fit_transform(X_train)

# Transform the test data using the ICA fitted on the training data
X_test_ica = ica.transform(X_test)

# Calculate kurtosis for each independent component (on the training data)
ica_kurtosis = kurtosis(X_train_ica, axis=0, fisher=False)

# Plotting kurtosis for each independent component
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(ica_kurtosis) + 1), ica_kurtosis)
plt.xlabel('Independent Component')
plt.ylabel('Kurtosis')
plt.title('Kurtosis of Independent Components (Training Set)')
plt.show()

# Display the kurtosis for each independent component
for i, k in enumerate(ica_kurtosis):
    print(f"Independent Component {i + 1}: {k:.4f} kurtosis")


# Test different numbers of components to maximize the absolute average kurtosis
max_components = X_train.shape[1] # Max number of components based on feature count
avg_kurtosis_values = []

for n_components in range(1, max_components + 1):
    ica = FastICA(n_components=n_components, random_state=seed)
    X_train_ica = ica.fit_transform(X_train)
    
    # Calculate kurtosis for each component
    ica_kurtosis = kurtosis(X_train_ica, axis=0, fisher=False)
    
    # Calculate the absolute average kurtosis
    avg_kurtosis = np.mean(np.abs(ica_kurtosis))
    avg_kurtosis_values.append(avg_kurtosis)

# Plotting the average absolute kurtosis for each number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_components + 1), avg_kurtosis_values, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Average Absolute Kurtosis')
plt.title('Average Absolute Kurtosis for Different Numbers of Components')
plt.show()

# Select the number of components that maximizes the absolute average kurtosis
optimal_components = np.argmax(avg_kurtosis_values) + 1
print(f'Optimal number of components: {optimal_components}')

# 10 number of components are enough to maximize the absolute average kurtosis

############ Random Projection ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define a range of n_components to test for dimensionality reduction
n_components_range = np.arange(1, 15)  # Make sure this is a valid range
reconstruction_errors = []
error_reductions = []

# Loop through each number of components
for n_components in n_components_range:
    # Apply Gaussian Random Projection on the training data
    grp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_train_grp = grp.fit_transform(X_train)
    
    # Reconstruct the training data using the pseudo-inverse of the projection matrix
    projection_matrix = grp.components_.T
    pseudo_inverse = np.linalg.pinv(projection_matrix)
    
    # Reconstruct the data
    X_train_reconstructed = np.dot(X_train_grp, pseudo_inverse[:X_train_grp.shape[1], :X_train.shape[1]])
    
    # Calculate Mean Squared Error
    reconstruction_error = np.mean((X_train - X_train_reconstructed) ** 2)
    reconstruction_errors.append(reconstruction_error)
    
    # Calculate error reduction from the previous step
    if n_components > 1:
        error_reduction = reconstruction_errors[-2] - reconstruction_error
        error_reductions.append(error_reduction)
    else:
        error_reductions.append(0)  # No error reduction for the first component

# Plot the reconstruction error
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, reconstruction_errors, marker='o', label='Reconstruction Error')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Reconstruction Error vs Number of Components')
plt.xticks(n_components_range)
plt.grid(True)
plt.show()

# Plot the error reduction per component
plt.figure(figsize=(10, 6))
plt.plot(n_components_range[1:], error_reductions[1:], marker='o', label='Error Reduction')
plt.xlabel('Number of Components')
plt.ylabel('Error Reduction (Δ MSE)')
plt.title('Error Reduction per Additional Component')
plt.xticks(n_components_range[1:])
plt.grid(True)
plt.show()

# 10 number of components are enough to minimize the reconstruction error

############# K-Means with PCA ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply PCA for dimensionality reduction on the training set
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)

# K-Means clustering on the PCA-reduced training data
range_n_clusters = range(2, 11)
pca_inertia_values = []
pca_silhouette_scores = []

for k in range_n_clusters:
    kmeans_pca = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
    kmeans_pca.fit(X_train_pca)
    pca_inertia_values.append(kmeans_pca.inertia_)
    silhouette_avg = silhouette_score(X_train_pca, kmeans_pca.labels_)
    pca_silhouette_scores.append(silhouette_avg)

# Plotting Inertia (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, pca_silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means on PCA-Reduced Data (Training Set)')
plt.xticks(range_n_clusters)

plt.tight_layout()
plt.show()

# Transform the test data using the PCA fitted on the training data
X_test_pca = pca.transform(X_test)

# Fit the K-Means model with the chosen number of clusters on the PCA-reduced training set
optimal_k = 9
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_pca)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Predict the clusters on the PCA-reduced test set
Y_pred = kmeans.predict(X_test_pca)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0253

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1182

############## Expectation Maximization with PCA ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply PCA for dimensionality reduction on the training set
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)

# Range of clusters to evaluate
range_n_clusters = range(2, 11)
bic_scores = []
gmm_silhouette_scores = []

for n_components in range_n_clusters:
    # Apply Gaussian Mixture Model (EM) to the PCA-reduced training data
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
    gmm.fit(X_train_pca)
    
    # Calculate BIC score and append it to the list
    bic = gmm.bic(X_train_pca)
    bic_scores.append(bic)
    
    # Predict cluster assignments
    cluster_labels = gmm.predict(X_train_pca)
    
    # Calculate silhouette score and append it to the list
    silhouette_avg = silhouette_score(X_train_pca, cluster_labels)
    gmm_silhouette_scores.append(silhouette_avg)

# Plotting BIC scores (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, bic_scores, marker='o')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.title('BIC Score for Gaussian Mixture on PCA-Reduced Data')
plt.xticks(range_n_clusters)
plt.tight_layout()
plt.show()

# Plotting Silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, gmm_silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Gaussian Mixture on PCA-Reduced Data')
plt.xticks(range_n_clusters)
plt.tight_layout()
plt.show()

# Transform the test data using the PCA fitted on the training data
X_test_pca = pca.transform(X_test)

# Fit the Gaussian Mixture Model (EM) with the chosen number of components on the PCA-reduced training set
optimal_components = 9
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full', random_state=seed)
gmm.fit(X_train_pca)

# Predict the clusters on the PCA-reduced test set
Y_pred = gmm.predict(X_test_pca)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0292

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") #0.1344

############### K-means with ICA ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply ICA for dimensionality reduction on the training set
ica = FastICA(n_components=5, random_state=seed)
X_train_ica = ica.fit_transform(X_train)

# Transform the test data using the ICA fitted on the training data
X_test_ica = ica.transform(X_test)

# K-Means clustering on the ICA-reduced training data
range_n_clusters = range(2, 10)
ica_inertia_values = []
ica_silhouette_scores = []

for k in range_n_clusters:
    kmeans_ica = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
    kmeans_ica.fit(X_train_ica)  # Fit K-Means on the training data reduced by ICA
    ica_inertia_values.append(kmeans_ica.inertia_)
    silhouette_avg = silhouette_score(X_train_ica, kmeans_ica.labels_)
    ica_silhouette_scores.append(silhouette_avg)

# Plotting Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, ica_silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means on ICA-Reduced Data (Training Set)')
plt.xticks(range_n_clusters)
plt.tight_layout()
plt.show()

# Fit the K-Means model with the chosen number of clusters on the ICA-reduced training set
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_ica)

# Predict the clusters on the ICA-reduced test set
Y_pred = kmeans.predict(X_test_ica)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0207

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1014

############### Expectation Maximization with ICA ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply ICA for dimensionality reduction on the training set
ica = FastICA(n_components=5, random_state=seed)  # Choose the number of components
X_train_ica = ica.fit_transform(X_train)

# Transform the test data using the ICA fitted on the training data
X_test_ica = ica.transform(X_test)

# EM Clustering (GMM) on the ICA-reduced training data
range_n_components = range(2, 10)
ica_bic_scores = []
ica_silhouette_scores = []

for n in range_n_components:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=seed)
    gmm.fit(X_train_ica)
    
    # Calculate BIC score and append it to the list
    bic = gmm.bic(X_train_ica)
    ica_bic_scores.append(bic)
    
    # Predict cluster assignments
    cluster_labels = gmm.predict(X_train_ica)
    
    # Calculate silhouette score and append it to the list
    silhouette_avg = silhouette_score(X_train_ica, cluster_labels)
    ica_silhouette_scores.append(silhouette_avg)

# Plotting BIC scores (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(range_n_components, ica_bic_scores, marker='o')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.title('BIC for GMM on ICA-Reduced Training Data')
plt.xticks(range_n_components)
plt.tight_layout()
plt.show()

# Transform the test data using the ICA fitted on the training data
X_test_ica = ica.transform(X_test)

# Fit the Gaussian Mixture Model (GMM) with the chosen number of components on the ICA-reduced training set
optimal_components = 8
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full', random_state=seed)
gmm.fit(X_train_ica)

# Predict the clusters on the ICA-reduced test set
Y_pred = gmm.predict(X_test_ica)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0366

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1426

############### K-means with Random Projection ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply Gaussian Random Projection for dimensionality reduction on the training set
grp = GaussianRandomProjection(n_components=8, random_state=seed)
X_train_grp = grp.fit_transform(X_train)

# Transform the test data using the RP fitted on the training data
X_test_grp = grp.transform(X_test)

# K-Means clustering on the RP-reduced training data
range_n_clusters = range(2, 11)
grp_inertia_values = []
grp_silhouette_scores = []

for k in range_n_clusters:
    kmeans_grp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
    kmeans_grp.fit(X_train_grp)  # Fit K-Means on the RP-reduced training data
    grp_inertia_values.append(kmeans_grp.inertia_)
    silhouette_avg = silhouette_score(X_train_grp, kmeans_grp.labels_)
    grp_silhouette_scores.append(silhouette_avg)

# Plotting Inertia (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, grp_silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means on Random Projection-Reduced Training Data')
plt.xticks(range_n_clusters)

plt.tight_layout()
plt.show()

# Transform the test data using the RP fitted on the training data
X_test_grp = grp.transform(X_test)

# Fit the K-Means model with the chosen number of clusters on the RP-reduced training set
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_grp)

# Predict the clusters on the RP-reduced test set
Y_pred = kmeans.predict(X_test_grp)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0627

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1667

############### Expectation Maximization with Random Projection ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply Gaussian Random Projection for dimensionality reduction on the training set
grp = GaussianRandomProjection(n_components=8, random_state=seed)
X_train_grp = grp.fit_transform(X_train)

# Transform the test data using the RP fitted on the training data
X_test_grp = grp.transform(X_test)

# EM Clustering (GMM) on the RP-reduced training data
range_n_components = range(2, 11)
grp_bic_scores = []
grp_silhouette_scores = []

for n in range_n_components:
    gmm = GaussianMixture(n_components=n, covariance_type='full', max_iter=300, random_state=seed)
    gmm.fit(X_train_grp)
    
    # Calculate BIC score and append it to the list
    bic = gmm.bic(X_train_grp)
    grp_bic_scores.append(bic)
    
    # Predict cluster assignments
    cluster_labels = gmm.predict(X_train_grp)
    
    # Calculate silhouette score and append it to the list
    silhouette_avg = silhouette_score(X_train_grp, cluster_labels)
    grp_silhouette_scores.append(silhouette_avg)

# Plotting BIC scores (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(range_n_components, grp_bic_scores, marker='o')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.title('BIC for GMM on Random Projection-Reduced Training Data')
plt.xticks(range_n_components)
plt.tight_layout()
plt.show()


# Transform the test data using the RP fitted on the training data
X_test_grp = grp.transform(X_test)

# Fit the Gaussian Mixture Model (GMM) with the chosen number of clusters on the RP-reduced training set
optimal_components = 7
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full', random_state=seed)
gmm.fit(X_train_grp)

# Predict the clusters on the RP-reduced test set
Y_pred = gmm.predict(X_test_grp)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0676

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.1405

################ Isomap ################
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# Define the range of n_neighbors values
n_neighbors_range = [5, 10, 15, 20, 30, 50, 100]

# Initialize LIL sparse matrix if needed (Example: a 1000x1000 matrix)
rows, cols = 1000, 1000
lil_mat = lil_matrix((rows, cols))

# Example of performing some efficient modifications to the LIL matrix
lil_mat[0, 1] = 1
lil_mat[2, 3] = 2
# Add more changes to lil_mat as needed

# Convert the matrix to CSR format for efficient computation after all modifications
csr_mat = lil_mat.tocsr()

# Initialize a list to store silhouette scores
silhouette_scores = []

# Loop over different n_neighbors values with fixed n_components=2
for n_neighbors in n_neighbors_range:
    # Set up Isomap with the current n_neighbors value and fixed n_components=2
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
    
    # Apply Isomap to the scaled training data (use your own X_train data)
    X_train_isomap = isomap.fit_transform(X_train)  # Assuming X_train is your training data
    
    # Check if at least two clusters exist in the projection
    if len(np.unique(Y_train)) > 1:  # Assuming Y_train contains your labels
        # Calculate Silhouette Score only if there are at least 2 clusters
        score = silhouette_score(X_train_isomap, Y_train)
        silhouette_scores.append(score)
        print(f'n_neighbors = {n_neighbors}, Silhouette Score = {score}')
    else:
        silhouette_scores.append(np.nan)  # If not enough clusters, append NaN
        print(f'n_neighbors = {n_neighbors}, Silhouette Score = Not Computed (only 1 cluster)')

# Plot the silhouette scores for different n_neighbors values
plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_range, silhouette_scores, marker='o', label='Isomap Silhouette Scores')

plt.title('Silhouette Score vs Number of Neighbors (Isomap)')
plt.xlabel('Number of Neighbors (n_neighbors)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

############ Isomap Components ############
# Define the range of n_components values to evaluate (from 2 to 11)
n_components_range = range(2, 12)

# Initialize LIL sparse matrix if needed (Example: a 1000x1000 matrix)
rows, cols = 1000, 1000
lil_mat = lil_matrix((rows, cols))

# Example of performing some efficient modifications to the LIL matrix
lil_mat[0, 1] = 1
lil_mat[2, 3] = 2
# Add more changes to lil_mat as needed

# Convert the matrix to CSR format for efficient computation after all modifications
csr_mat = lil_mat.tocsr()

# Initialize a list to store silhouette scores
silhouette_scores = []

# Loop over different n_components values with fixed n_neighbors=30
for n_components in n_components_range:
    # Set up Isomap with the current n_components value and fixed n_neighbors=30
    isomap = Isomap(n_neighbors=30, n_components=n_components)
    
    # Apply Isomap to the scaled training data (use your own X_train data)
    X_train_isomap = isomap.fit_transform(X_train)  # Assuming X_train is your training data
    
    # Check if at least two clusters exist in the projection
    if len(np.unique(Y_train)) > 1:  # Assuming Y_train contains your labels
        # Calculate Silhouette Score only if there are at least 2 clusters
        score = silhouette_score(X_train_isomap, Y_train)
        silhouette_scores.append(score)
        print(f'n_components = {n_components}, Silhouette Score = {score}')
    else:
        silhouette_scores.append(np.nan)  # If not enough clusters, append NaN
        print(f'n_components = {n_components}, Silhouette Score = Not Computed (only 1 cluster)')

# Plot the silhouette scores for different n_components values
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, silhouette_scores, marker='o', label='Isomap Silhouette Scores')

plt.title('Silhouette Score vs Number of Components (Isomap with 30 Neighbors)')
plt.xlabel('Number of Components (n_components)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

########## Isomap Visualization ##########
# Initialize LIL sparse matrix if needed (Example: a 1000x1000 matrix)
rows, cols = 1000, 1000
lil_mat = lil_matrix((rows, cols))

# Example of performing some efficient modifications to the LIL matrix
lil_mat[0, 1] = 1
lil_mat[2, 3] = 2
# Add more changes to lil_mat as needed

# Convert the matrix to CSR format for efficient computation after all modifications
csr_mat = lil_mat.tocsr()

# Set up Isomap with n_neighbors=30 and n_components=2
isomap = Isomap(n_neighbors=30, n_components=2)

# Apply Isomap to the scaled training data (use your own X_train data)
X_train_isomap = isomap.fit_transform(X_train)  # Assuming X_train is your training data

# Check if at least two clusters exist in the projection
if len(np.unique(Y_train)) > 1:  # Assuming Y_train contains your labels
    # Calculate Silhouette Score only if there are at least 2 clusters
    score = silhouette_score(X_train_isomap, Y_train)
    print(f'n_neighbors = 30, Silhouette Score = {score}')
else:
    score = np.nan  # If not enough clusters, append NaN
    print(f'n_neighbors = 30, Silhouette Score = Not Computed (only 1 cluster)')

# Plot the Isomap results with color coding based on the 'Y_train' labels (assuming binary classification)
plt.figure(figsize=(10, 7))
plt.scatter(X_train_isomap[Y_train == 0, 0], X_train_isomap[Y_train == 0, 1], label='Non-Churner', alpha=0.6, s=10, c='blue')
plt.scatter(X_train_isomap[Y_train == 1, 0], X_train_isomap[Y_train == 1, 1], label='Churner', alpha=0.6, s=10, c='orange')

plt.title('Isomap Projection with 30 Neighbors and 2 Components')
plt.xlabel('Isomap Component 1')
plt.ylabel('Isomap Component 2')
plt.legend()
plt.grid(True)
plt.show()
