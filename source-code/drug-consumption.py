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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import Isomap
from scipy.sparse import lil_matrix


# Set the seed for reproducibility
seed = 42
np.random.seed(seed)         # For NumPy operations
random.seed(seed)            # For Python's built-in random operations

# fetch dataset - drug consumption (quantified)
drug_consumption_quantified = fetch_ucirepo(id=373) 
  
# data (as pandas dataframes)
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets 
  
# metadata 
# print(drug_consumption_quantified.metadata) 
  
# variable information 
# print(drug_consumption_quantified.variables) 

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
columns_X = ["age","gender","education","country","ethnicity","nscore","escore","oscore","ascore",
             "cscore","impulsive","ss"]
X_scaled = pd.DataFrame(X_scaled, columns=columns_X)

# Create a binary classification problem
class_mapping = {
    'CL0': 'Non-user',
    'CL1': 'Non-user',
    'CL2': 'User',
    'CL3': 'User',
    'CL4': 'User',
    'CL5': 'User',
    'CL6': 'User'
}

# Apply the mapping to each target column
y.loc[:, 'cannabis'] = y['cannabis'].map(class_mapping)
Y_Cannabis = y['cannabis']
Y = Y_Cannabis.values.ravel()
Y = np.where(Y == 'Non-user', 0, 1)

pd.set_option('display.max_columns', None)  # Show all columns
print(Y)
pd.reset_option('display.max_columns')

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
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)



# Predict the clusters on the test set
Y_pred = kmeans.predict(X_test)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.1794

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.2284

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
optimal_component = 2
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)


# Predict cluster assignments on the test set
Y_pred = gmm.predict(X_test)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.1077

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") #0.1162


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

# 7 number of components are enough to explain 80% of the variance

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
max_components = X_train.shape[1]  # Max number of components based on feature count
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

# 5 number of components are enough to maximize the absolute average kurtosis

############ Random Projection ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define a range of n_components to test for dimensionality reduction
n_components_range = range(1, 15)
reconstruction_errors = []

for n_components in n_components_range:
    # Apply Gaussian Random Projection on the training data
    grp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_train_grp = grp.fit_transform(X_train)
    
    # Reconstruct the training data using the pseudo-inverse of the projection matrix
    projection_matrix = grp.components_.T
    pseudo_inverse = np.linalg.pinv(projection_matrix)
    
    # Ensure that the reconstruction results in the same number of features as the original X_train
    X_train_reconstructed = np.dot(X_train_grp, pseudo_inverse[:X_train_grp.shape[1], :X_train.shape[1]])
    
    # Calculate Mean Squared Error manually for the training set
    reconstruction_error = np.mean((X_train - X_train_reconstructed) ** 2)
    reconstruction_errors.append(reconstruction_error)

# Plot the reconstruction error against the number of components
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, reconstruction_errors, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Reconstruction Error vs Number of Components for Random Projection (Training Set)')
# Set the x-axis ticks to integers
plt.xticks(n_components_range)
plt.grid(True)
plt.show()

# 8 number of components are enough to minimize the reconstruction error

############# K-Means with PCA ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply PCA for dimensionality reduction on the training set
pca = PCA(n_components=7)
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
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_pca)

# Predict the clusters on the PCA-reduced test set
Y_pred = kmeans.predict(X_test_pca)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.1748

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.2258

############## Expectation Maximization with PCA ##############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply PCA for dimensionality reduction on the training set
pca = PCA(n_components=7)
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
optimal_components = 5
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full', random_state=seed)
gmm.fit(X_train_pca)

# Predict the clusters on the PCA-reduced test set
Y_pred = gmm.predict(X_test_pca)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0675

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") #0.1212

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
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_ica)

# Predict the clusters on the ICA-reduced test set
Y_pred = kmeans.predict(X_test_ica)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0463

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.0958

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
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0471

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.0996

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
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train_grp)

# Predict the clusters on the RP-reduced test set
Y_pred = kmeans.predict(X_test_grp)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.2039

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.2106

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
optimal_components = 4
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full', random_state=seed)
gmm.fit(X_train_grp)

# Predict the clusters on the RP-reduced test set
Y_pred = gmm.predict(X_test_grp)

# Calculate Adjusted Rand Index (ARI)
ari_score = adjusted_rand_score(Y_test, Y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}") # 0.0242

# Calculate Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(Y_test, Y_pred)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}") # 0.0239

############### Extra Credit: Isomap ###############
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
    isomap = Isomap(n_neighbors=50, n_components=n_components)
    
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

# Set up Isomap with n_neighbors=50 and n_components=2
isomap = Isomap(n_neighbors=50, n_components=2)

# Apply Isomap to the scaled training data (use your own X_train data)
X_train_isomap = isomap.fit_transform(X_train)  # Assuming X_train is your training data

# Check if at least two clusters exist in the projection
if len(np.unique(Y_train)) > 1:  # Assuming Y_train contains your labels
    # Calculate Silhouette Score only if there are at least 2 clusters
    score = silhouette_score(X_train_isomap, Y_train)
    print(f'n_neighbors = 50, Silhouette Score = {score}')
else:
    score = np.nan  # If not enough clusters, append NaN
    print(f'n_neighbors = 50, Silhouette Score = Not Computed (only 1 cluster)')

# Plot the Isomap results with color coding based on the 'Y_train' labels (assuming binary classification)
plt.figure(figsize=(10, 7))
plt.scatter(X_train_isomap[Y_train == 0, 0], X_train_isomap[Y_train == 0, 1], label='Non-user', alpha=0.6, s=10, c='blue')
plt.scatter(X_train_isomap[Y_train == 1, 0], X_train_isomap[Y_train == 1, 1], label='User', alpha=0.6, s=10, c='orange')

plt.title('Isomap Projection with 50 Neighbors and 2 Components')
plt.xlabel('Isomap Component 1')
plt.ylabel('Isomap Component 2')
plt.legend()
plt.grid(True)
plt.show()







######## Selected Neural Network Model #######
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score
import time

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the neural network model without SMOTE
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam', 
                      random_state=seed, batch_size=128, learning_rate_init=0.0001,
                      alpha=0.001, max_iter=500)

# Create a pipeline with only the neural network model
pipeline = make_pipeline(model)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred = cross_val_predict(pipeline, X_train, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Train the model on the full training set
start_time = time.time()
pipeline.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred = pipeline.predict(X_test)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations = model.n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted = f1_score(Y_test, Y_test_pred, average='weighted')

# Print out timing, convergence, and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Training Time for the MLP model: {time_mlp_training} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Number of Iterations for MLP Convergence: {mlp_iterations}")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

################# Neural Network Raw Data Results ############
# Cross-validation Prediction Time: 8.7365 seconds
# Training Time for the MLP model: 1.72 seconds
# Prediction Time for the MLP model: 0.0074 seconds
# Number of Iterations for MLP Convergence: 251
# Cross-validated Training F1 Score (Weighted): 0.8057
# Test F1 Score (Weighted): 0.8269

############## Neural Network Learning Curve Base Data ############
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the neural network model without SMOTE
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam',
                      random_state=seed, batch_size=128, learning_rate_init=0.0001,
                      alpha=0.001, max_iter=1, warm_start=True)  # Set max_iter=1 for one epoch at a time

# Initialize lists to track F1 scores for training and test sets
train_f1_scores_raw_data = []  # Renamed
test_f1_scores_raw_data = []   # Renamed
epochs = 100  # Number of epochs you want to track

# Train incrementally and track F1 scores after each epoch
for epoch in range(epochs):
    # Use partial_fit for incremental learning
    if epoch == 0:
        # For the first iteration, we need to specify the classes for partial_fit
        model.partial_fit(X_train, Y_train, classes=np.unique(Y))
    else:
        model.partial_fit(X_train, Y_train)
    
    # Predict on the training set
    Y_train_pred = model.predict(X_train)
    train_f1 = f1_score(Y_train, Y_train_pred, average='weighted')
    train_f1_scores_raw_data.append(train_f1)  # Updated name
    
    # Predict on the test set
    Y_test_pred = model.predict(X_test)
    test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
    test_f1_scores_raw_data.append(test_f1)  # Updated name

# Plot the learning curve (F1 score vs. Epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_f1_scores_raw_data, label='Training F1 Score', color='r', marker='o')
plt.plot(range(1, epochs + 1), test_f1_scores_raw_data, label='Test F1 Score', color='g', marker='o')
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve (F1 Score vs Epochs)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


########### PCA NN Grid Search ############
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time

# Define the parameter grid for the number of hidden layers and their sizes
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Set up PCA with the desired number of components
pca = PCA(n_components=7)  # Adjust the number of components if needed

# Define the pipeline: PCA + MLP
pipeline_pca = Pipeline([
    ('pca', pca),
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed, 
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(pipeline_pca, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

# Measure wall clock time for the grid search
start_time = time.time()
grid_search.fit(X_train, Y_train)
end_time = time.time()

# Wall clock time for grid search
grid_search_time = round(end_time - start_time, 4)

# Best model and its score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Test the best model on the test set
Y_test_pred_pca = grid_search.predict(X_test)

# Calculate weighted F1 score on the test data using the best model
test_f1_weighted_pca = f1_score(Y_test, Y_test_pred_pca, average='weighted')

# Print results
# Best Parameters: {'mlp__hidden_layer_sizes': (64, 32)}
# Best Cross-validated F1 Score (weighted): 0.8079903740309007
# Grid Search Time: 16.6361 seconds
# Test F1 Score (weighted) with best model: 0.8377354498446675

############ PCA Neural Network Model ############
# Measure wall clock time for PCA + Neural Network Model without SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score
import time

# Set up PCA with desired number of components
pca = PCA(n_components=7)  # Adjust the number of components as needed

# Update the pipeline to include only PCA before the neural network, without SMOTE
pipeline_pca = Pipeline([
    ('pca', pca),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data with PCA included
start_time = time.time()
Y_train_pred_pca = cross_val_predict(pipeline_pca, X_train, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction_pca = round(end_time - start_time, 4)

# Train the model on the full PCA-transformed training set
start_time = time.time()
pipeline_pca.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training_pca = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_pca = pipeline_pca.predict(X_test)
end_time = time.time()
time_mlp_prediction_pca = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge with PCA applied
mlp_iterations_pca = pipeline_pca.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted_pca = f1_score(Y_train, Y_train_pred_pca, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted_pca = f1_score(Y_test, Y_test_pred_pca, average='weighted')

# Print out timing, convergence, and performance metrics for PCA pipeline without SMOTE
print(f"Cross-validation Prediction Time with PCA: {time_cv_prediction_pca} seconds")
print(f"Training Time for the MLP model with PCA: {time_mlp_training_pca} seconds")
print(f"Prediction Time for the MLP model with PCA: {time_mlp_prediction_pca} seconds")
print(f"Number of Iterations for MLP Convergence with PCA: {mlp_iterations_pca}")
print(f"Cross-validated Training F1 Score with PCA (Weighted): {train_f1_weighted_pca:.4f}")
print(f"Test F1 Score with PCA (Weighted): {test_f1_weighted_pca:.4f}")

################# Neural Network PCA Results ############
# Cross-validation Prediction Time with PCA: 4.2371 seconds
# Training Time for the MLP model with PCA: 1.893 seconds
# Prediction Time for the MLP model with PCA: 0.0082 seconds
# Number of Iterations for MLP Convergence with PCA: 426
# Cross-validated Training F1 Score with PCA (Weighted): 0.8063
# Test F1 Score with PCA (Weighted): 0.8377

############# PCA Neural Network Model Learning Curve ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set up PCA with desired number of components
pca = PCA(n_components=7)

# Define the neural network model without SMOTE, with PCA as part of the pipeline
model_pca = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=1, warm_start=True)  # Max_iter=1 for one epoch

# Create a pipeline with PCA and the neural network model
pipeline_pca = Pipeline([
    ('pca', pca),
    ('mlp', model_pca)
])

# Initialize lists to track F1 scores for training and test sets
train_f1_scores_pca_data = []  # Renamed
test_f1_scores_pca_data = []   # Renamed
epochs = 100  # Number of epochs you want to track

# Train incrementally and track F1 scores after each epoch
for epoch in range(epochs):
    # Train incrementally with partial_fit
    if epoch == 0:
        # Fit PCA first, then apply partial_fit to the neural network
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        # For the first epoch, use partial_fit with the class labels
        model_pca.partial_fit(X_train_pca, Y_train, classes=np.unique(Y_train))
    else:
        # Use partial_fit on PCA-transformed training data
        model_pca.partial_fit(X_train_pca, Y_train)
    
    # Predict on the training set
    Y_train_pred_pca = model_pca.predict(X_train_pca)
    train_f1_pca = f1_score(Y_train, Y_train_pred_pca, average='weighted')
    train_f1_scores_pca_data.append(train_f1_pca)  # Updated name
    
    # Predict on the test set
    Y_test_pred_pca = model_pca.predict(X_test_pca)
    test_f1_pca = f1_score(Y_test, Y_test_pred_pca, average='weighted')
    test_f1_scores_pca_data.append(test_f1_pca)  # Updated name

# Plot the learning curve (F1 score vs. Epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_f1_scores_pca_data, label='Training F1 Score (PCA)', color='r', marker='o')
plt.plot(range(1, epochs + 1), test_f1_scores_pca_data, label='Test F1 Score (PCA)', color='g', marker='o')
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve (F1 Score vs Epochs) for PCA + Neural Network')
plt.legend(loc='best')
plt.grid(True)
plt.show()

############### ICA NN Grid Search ############
# Define the parameter grid for the number of hidden layers and their sizes
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Set up ICA with the desired number of components
ica = FastICA(n_components=5, random_state=seed)

# Create the pipeline: ICA + MLP
pipeline_ica = Pipeline([
    ('ica', ica),
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed, 
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform grid search with 5-fold cross-validation
grid_search_ica = GridSearchCV(pipeline_ica, param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1)

# Measure wall clock time for the grid search
start_time = time.time()
grid_search_ica.fit(X_train, Y_train)
end_time = time.time()

# Wall clock time for grid search
grid_search_time_ica = round(end_time - start_time, 4)

# Best model and its score
best_params_ica = grid_search_ica.best_params_
best_score_ica = grid_search_ica.best_score_

# Test the best model on the test set
Y_test_pred_ica = grid_search_ica.predict(X_test)

# Calculate weighted F1 score on the test data using the best model
test_f1_weighted_ica = f1_score(Y_test, Y_test_pred_ica, average='weighted')

# Print results
print(f"Best Parameters for ICA: {best_params_ica}")
print(f"Best Cross-validated F1 Score (weighted) with ICA: {best_score_ica}")
print(f"Grid Search Time for ICA: {grid_search_time_ica} seconds")
print(f"Test F1 Score (weighted) with ICA: {test_f1_weighted_ica}")

# Best Parameters for ICA: {'mlp__hidden_layer_sizes': (64,)}
# Best Cross-validated F1 Score (weighted) with ICA: 0.8060836090945518
# Grid Search Time for ICA: 43.2049 seconds
# Test F1 Score (weighted) with ICA: 0.8164251821263944

############## ICA Neural Network Model #############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Necessary imports for ICA
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline

# Set up ICA with desired number of components
ica = FastICA(n_components=5, random_state=seed)  # Adjust the number of components as needed

# Create a pipeline with ICA and the neural network model, without SMOTE
pipeline_ica = Pipeline([
    ('ica', ica),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64,), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data with ICA included
start_time = time.time()
Y_train_pred_ica = cross_val_predict(pipeline_ica, X_train, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction_ica = round(end_time - start_time, 4)

# Train the model on the full ICA-transformed training set
start_time = time.time()
pipeline_ica.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training_ica = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_ica = pipeline_ica.predict(X_test)
end_time = time.time()
time_mlp_prediction_ica = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge with ICA applied
mlp_iterations_ica = pipeline_ica.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted_ica = f1_score(Y_train, Y_train_pred_ica, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted_ica = f1_score(Y_test, Y_test_pred_ica, average='weighted')

# Print out timing, convergence, and performance metrics for ICA pipeline without SMOTE
print(f"Cross-validation Prediction Time with ICA: {time_cv_prediction_ica} seconds")
print(f"Training Time for the MLP model with ICA: {time_mlp_training_ica} seconds")
print(f"Prediction Time for the MLP model with ICA: {time_mlp_prediction_ica} seconds")
print(f"Number of Iterations for MLP Convergence with ICA: {mlp_iterations_ica}")
print(f"Cross-validated Training F1 Score with ICA (Weighted): {train_f1_weighted_ica:.4f}")
print(f"Test F1 Score with ICA (Weighted): {test_f1_weighted_ica:.4f}")

# Cross-validation Prediction Time with ICA: 2.6073 seconds
# Training Time for the MLP model with ICA: 0.528 seconds
# Prediction Time for the MLP model with ICA: 0.0069 seconds
# Number of Iterations for MLP Convergence with ICA: 177
# Cross-validated Training F1 Score with ICA (Weighted): 0.8062
# Test F1 Score with ICA (Weighted): 0.8164

################ ICA Neural Network Model Learning Curve ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set up ICA with the desired number of components
ica = FastICA(n_components=6, random_state=seed)

# Define the neural network model without SMOTE
model_ica = MLPClassifier(hidden_layer_sizes=(64,), activation='tanh', solver='adam',
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=1, warm_start=True)  # Max_iter=1 for one epoch at a time

# Create a pipeline with ICA and the neural network model
pipeline_ica = Pipeline([
    ('ica', ica),
    ('mlp', model_ica)
])

# Initialize lists to track F1 scores for training and test sets
train_f1_scores_ica_data = []  # Renamed
test_f1_scores_ica_data = []   # Renamed
epochs = 100  # Number of epochs you want to track

# Transform the data with ICA first, since we need to apply partial_fit separately
X_train_ica = ica.fit_transform(X_train)
X_test_ica = ica.transform(X_test)

# Train incrementally and track F1 scores after each epoch
for epoch in range(epochs):
    # Train incrementally with partial_fit
    if epoch == 0:
        # For the first epoch, use partial_fit with the class labels
        model_ica.partial_fit(X_train_ica, Y_train, classes=np.unique(Y_train))
    else:
        model_ica.partial_fit(X_train_ica, Y_train)
    
    # Predict on the training set
    Y_train_pred_ica = model_ica.predict(X_train_ica)
    train_f1_ica = f1_score(Y_train, Y_train_pred_ica, average='weighted')
    train_f1_scores_ica_data.append(train_f1_ica)  # Updated name
    
    # Predict on the test set
    Y_test_pred_ica = model_ica.predict(X_test_ica)
    test_f1_ica = f1_score(Y_test, Y_test_pred_ica, average='weighted')
    test_f1_scores_ica_data.append(test_f1_ica)  # Updated name

# Plot the learning curve (F1 score vs. Epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_f1_scores_ica_data, label='Training F1 Score (ICA)', color='r', marker='o')
plt.plot(range(1, epochs + 1), test_f1_scores_ica_data, label='Test F1 Score (ICA)', color='g', marker='o')
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve (F1 Score vs Epochs) for ICA + Neural Network')
plt.legend(loc='best')
plt.grid(True)
plt.show()

############### RP NN Grid Search ############
# Define the parameter grid for the number of hidden layers and their sizes
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Set up Random Projection (RP) with the desired number of components
grp = GaussianRandomProjection(n_components=6, random_state=seed)  # Adjust the number of components if needed

# Create a pipeline: RP + MLP
pipeline_grp = Pipeline([
    ('grp', grp),
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed, 
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform grid search with 5-fold cross-validation
grid_search_grp = GridSearchCV(pipeline_grp, param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1)

# Measure wall clock time for the grid search
start_time = time.time()
grid_search_grp.fit(X_train, Y_train)
end_time = time.time()

# Wall clock time for grid search
grid_search_time_grp = round(end_time - start_time, 4)

# Best model and its score
best_params_grp = grid_search_grp.best_params_
best_score_grp = grid_search_grp.best_score_

# Test the best model on the test set
Y_test_pred_grp = grid_search_grp.predict(X_test)

# Calculate weighted F1 score on the test data using the best model
test_f1_weighted_grp = f1_score(Y_test, Y_test_pred_grp, average='weighted')

# Print results
print(f"Best Parameters for Random Projection: {best_params_grp}")
print(f"Best Cross-validated F1 Score (weighted) with Random Projection: {best_score_grp}")
print(f"Grid Search Time for Random Projection: {grid_search_time_grp} seconds")
print(f"Test F1 Score (weighted) with Random Projection: {test_f1_weighted_grp}")

# Best Parameters for Random Projection: {'mlp__hidden_layer_sizes': (32,)}
# Best Cross-validated F1 Score (weighted) with Random Projection: 0.7966086217231604
# Grid Search Time for Random Projection: 12.0413 seconds
# Test F1 Score (weighted) with Random Projection: 0.7744652237552847

############ RP Neural Network Model ############
# Necessary imports for Random Projection
# Set up Random Projection with desired number of components
grp = GaussianRandomProjection(n_components=6, random_state=seed)  # Adjust the number of components as needed

# Create a pipeline with Random Projection and the neural network model, without SMOTE
pipeline_grp = Pipeline([
    ('grp', grp),
    ('mlp', MLPClassifier(hidden_layer_sizes=(32,), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data with Random Projection included
start_time = time.time()
Y_train_pred_grp = cross_val_predict(pipeline_grp, X_train, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction_grp = round(end_time - start_time, 4)

# Train the model on the full Random Projection-transformed training set
start_time = time.time()
pipeline_grp.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training_grp = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_grp = pipeline_grp.predict(X_test)
end_time = time.time()
time_mlp_prediction_grp = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge with Random Projection applied
mlp_iterations_grp = pipeline_grp.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted_grp = f1_score(Y_train, Y_train_pred_grp, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted_grp = f1_score(Y_test, Y_test_pred_grp, average='weighted')

# Print out timing, convergence, and performance metrics for Random Projection pipeline without SMOTE
print(f"Cross-validation Prediction Time with Random Projection: {time_cv_prediction_grp} seconds")
print(f"Training Time for the MLP model with Random Projection: {time_mlp_training_grp} seconds")
print(f"Prediction Time for the MLP model with Random Projection: {time_mlp_prediction_grp} seconds")
print(f"Number of Iterations for MLP Convergence with Random Projection: {mlp_iterations_grp}")
print(f"Cross-validated Training F1 Score with Random Projection (Weighted): {train_f1_weighted_grp:.4f}")
print(f"Test F1 Score with Random Projection (Weighted): {test_f1_weighted_grp:.4f}")

# Cross-validation Prediction Time with Random Projection: 1.5787 seconds
# Training Time for the MLP model with Random Projection: 0.3287 seconds
# Prediction Time for the MLP model with Random Projection: 0.0068 seconds
# Number of Iterations for MLP Convergence with Random Projection: 140
# Cross-validated Training F1 Score with Random Projection (Weighted): 0.7967
# Test F1 Score with Random Projection (Weighted): 0.7745

################# RP Neural Network Model Learning Curve ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set up Random Projection with the desired number of components
grp = GaussianRandomProjection(n_components=6, random_state=seed)

# Define the neural network model without SMOTE
model_grp = MLPClassifier(hidden_layer_sizes=(32,), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=1, warm_start=True)  # Max_iter=1 for one epoch at a time

# Create a pipeline with Random Projection and the neural network model
pipeline_grp = Pipeline([
    ('grp', grp),
    ('mlp', model_grp)
])

# Initialize lists to track F1 scores for training and test sets
train_f1_scores_grp_data = []  # Renamed
test_f1_scores_grp_data = []   # Renamed
epochs = 100  # Number of epochs you want to track

# Transform the data with Random Projection first, since we need to apply partial_fit separately
X_train_grp = grp.fit_transform(X_train)
X_test_grp = grp.transform(X_test)

# Train incrementally and track F1 scores after each epoch
for epoch in range(epochs):
    # Train incrementally with partial_fit
    if epoch == 0:
        # For the first epoch, use partial_fit with the class labels
        model_grp.partial_fit(X_train_grp, Y_train, classes=np.unique(Y_train))
    else:
        model_grp.partial_fit(X_train_grp, Y_train)
    
    # Predict on the training set
    Y_train_pred_grp = model_grp.predict(X_train_grp)
    train_f1_grp = f1_score(Y_train, Y_train_pred_grp, average='weighted')
    train_f1_scores_grp_data.append(train_f1_grp)  # Updated name
    
    # Predict on the test set
    Y_test_pred_grp = model_grp.predict(X_test_grp)
    test_f1_grp = f1_score(Y_test, Y_test_pred_grp, average='weighted')
    test_f1_scores_grp_data.append(test_f1_grp)  # Updated name

# Plot the learning curve (F1 score vs. Epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_f1_scores_grp_data, label='Training F1 Score (RP)', color='r', marker='o')
plt.plot(range(1, epochs + 1), test_f1_scores_grp_data, label='Test F1 Score (RP)', color='g', marker='o')
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve (F1 Score vs Epochs) for RP + Neural Network')
plt.legend(loc='best')
plt.grid(True)
plt.show()


################# Isomap NN Grid Search ############
# Define the parameter grid for the number of hidden layers and their sizes
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Set up Isomap with 50 neighbors and 2 components
isomap = Isomap(n_neighbors=50, n_components=2)

# Create a pipeline: Isomap + MLP
pipeline_isomap = Pipeline([
    ('isomap', isomap),
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed, 
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform grid search with 5-fold cross-validation
grid_search_isomap = GridSearchCV(pipeline_isomap, param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1)

# Measure wall clock time for the grid search
start_time = time.time()
grid_search_isomap.fit(X_train, Y_train)
end_time = time.time()

# Wall clock time for grid search
grid_search_time_isomap = round(end_time - start_time, 4)

# Best model and its score
best_params_isomap = grid_search_isomap.best_params_
best_score_isomap = grid_search_isomap.best_score_

# Test the best model on the test set
Y_test_pred_isomap = grid_search_isomap.predict(X_test)

# Calculate weighted F1 score on the test data using the best model
test_f1_weighted_isomap = f1_score(Y_test, Y_test_pred_isomap, average='weighted')

# Print results
print(f"Best Parameters for Isomap: {best_params_isomap}")
print(f"Best Cross-validated F1 Score (weighted) with Isomap: {best_score_isomap}")
print(f"Grid Search Time for Isomap: {grid_search_time_isomap} seconds")
print(f"Test F1 Score (weighted) with Isomap: {test_f1_weighted_isomap}")

# Best Parameters for Isomap: {'mlp__hidden_layer_sizes': (64, 32, 16)}
# Best Cross-validated F1 Score (weighted) with Isomap: 0.7848504540825794
# Grid Search Time for Isomap: 22.7938 seconds
# Test F1 Score (weighted) with Isomap: 0.7942216522239169

################ Isomap Neural Network Model ############
# Set up Isomap with the desired number of neighbors and components
isomap = Isomap(n_neighbors=50, n_components=2)  # Adjust neighbors and components if needed

# Create a pipeline with Isomap and the neural network model, without SMOTE
pipeline_isomap = Pipeline([
    ('isomap', isomap),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='adam', 
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500))
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data with Isomap included
start_time = time.time()
Y_train_pred_isomap = cross_val_predict(pipeline_isomap, X_train, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction_isomap = round(end_time - start_time, 4)

# Train the model on the full Isomap-transformed training set
start_time = time.time()
pipeline_isomap.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training_isomap = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_isomap = pipeline_isomap.predict(X_test)
end_time = time.time()
time_mlp_prediction_isomap = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge with Isomap applied
mlp_iterations_isomap = pipeline_isomap.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted_isomap = f1_score(Y_train, Y_train_pred_isomap, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted_isomap = f1_score(Y_test, Y_test_pred_isomap, average='weighted')

# Print out timing, convergence, and performance metrics for Isomap pipeline without SMOTE
print(f"Cross-validation Prediction Time with Isomap: {time_cv_prediction_isomap} seconds")
print(f"Training Time for the MLP model with Isomap: {time_mlp_training_isomap} seconds")
print(f"Prediction Time for the MLP model with Isomap: {time_mlp_prediction_isomap} seconds")
print(f"Number of Iterations for MLP Convergence with Isomap: {mlp_iterations_isomap}")
print(f"Cross-validated Training F1 Score with Isomap (Weighted): {train_f1_weighted_isomap:.4f}")
print(f"Test F1 Score with Isomap (Weighted): {test_f1_weighted_isomap:.4f}")


# Cross-validation Prediction Time with Isomap: 8.4931 seconds
# Training Time for the MLP model with Isomap: 3.818 seconds
# Prediction Time for the MLP model with Isomap: 0.1241 seconds
# Number of Iterations for MLP Convergence with Isomap: 62
# Cross-validated Training F1 Score with Isomap (Weighted): 0.7847
# Test F1 Score with Isomap (Weighted): 0.7942


############### Isomap Neural Network Model Learning Curve ############
################# Isomap Neural Network Model Learning Curve ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set up Isomap with the desired number of neighbors and components
isomap = Isomap(n_neighbors=50, n_components=2)

# Define the neural network model without SMOTE
model_isomap = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='adam', 
                             random_state=seed, batch_size=128, learning_rate_init=0.0001,
                             alpha=0.001, max_iter=1, warm_start=True)  # Max_iter=1 for one epoch at a time

# Create a pipeline with Isomap and the neural network model
pipeline_isomap = Pipeline([
    ('isomap', isomap),
    ('mlp', model_isomap)
])

# Initialize lists to track F1 scores for training and test sets
train_f1_scores_isomap_data = []  # Renamed
test_f1_scores_isomap_data = []   # Renamed
epochs = 100  # Number of epochs you want to track

# Transform the data with Isomap first, since we need to apply partial_fit separately
X_train_isomap = isomap.fit_transform(X_train)
X_test_isomap = isomap.transform(X_test)

# Train incrementally and track F1 scores after each epoch
for epoch in range(epochs):
    # Train incrementally with partial_fit
    if epoch == 0:
        # For the first epoch, use partial_fit with the class labels
        model_isomap.partial_fit(X_train_isomap, Y_train, classes=np.unique(Y_train))
    else:
        model_isomap.partial_fit(X_train_isomap, Y_train)
    
    # Predict on the training set
    Y_train_pred_isomap = model_isomap.predict(X_train_isomap)
    train_f1_isomap = f1_score(Y_train, Y_train_pred_isomap, average='weighted')
    train_f1_scores_isomap_data.append(train_f1_isomap)  # Updated name
    
    # Predict on the test set
    Y_test_pred_isomap = model_isomap.predict(X_test_isomap)
    test_f1_isomap = f1_score(Y_test, Y_test_pred_isomap, average='weighted')
    test_f1_scores_isomap_data.append(test_f1_isomap)  # Updated name

# Plot the learning curve (F1 score vs. Epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_f1_scores_isomap_data, label='Training F1 Score (Isomap)', color='r', marker='o')
plt.plot(range(1, epochs + 1), test_f1_scores_isomap_data, label='Test F1 Score (Isomap)', color='g', marker='o')
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve (F1 Score vs Epochs) for Isomap + Neural Network')
plt.legend(loc='best')
plt.grid(True)
plt.show()

############### Training Learning Curve for Raw Data, PCA, ICA, RP, and Isomap ############
# Plot the training F1 scores for each dataset in one graph
plt.figure(figsize=(10, 6))

# Plot each dataset's F1 scores over epochs with smaller points (markersize)
plt.plot(range(1, epochs + 1), train_f1_scores_raw_data, label='Original Data', color='b', marker='o', markersize=3)
plt.plot(range(1, epochs + 1), train_f1_scores_pca_data, label='PCA', color='r', marker='x', markersize=3)
plt.plot(range(1, epochs + 1), train_f1_scores_ica_data, label='ICA', color='g', marker='s', markersize=3)
plt.plot(range(1, epochs + 1), train_f1_scores_grp_data, label='RP', color='m', marker='^', markersize=3)
plt.plot(range(1, epochs + 1), train_f1_scores_isomap_data, label='Isomap', color='y', marker='d', markersize=3)

# Label the axes
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Training F1 Score')

# Add a title
plt.title('Training F1 Score vs Epoch for Different Dimensionality Reduction Techniques')

# Show legend
plt.legend(loc='best')

# Add grid for clarity
plt.grid(True)

# Display the plot
plt.show()

############### Test Learning Curve for Raw Data, PCA, ICA, RP, and Isomap ############
# Plot the test F1 scores for each dataset in one graph
plt.figure(figsize=(10, 6))

# Plot each dataset's test F1 scores over epochs with smaller points (markersize=3)
plt.plot(range(1, epochs + 1), test_f1_scores_raw_data, label='Original Data', color='b', marker='o', markersize=3)
plt.plot(range(1, epochs + 1), test_f1_scores_pca_data, label='PCA', color='r', marker='x', markersize=3)
plt.plot(range(1, epochs + 1), test_f1_scores_ica_data, label='ICA', color='g', marker='s', markersize=3)
plt.plot(range(1, epochs + 1), test_f1_scores_grp_data, label='RP', color='m', marker='^', markersize=3)
plt.plot(range(1, epochs + 1), test_f1_scores_isomap_data, label='Isomap', color='y', marker='d', markersize=3)

# Label the axes
plt.xlabel('Epoch (Number of Iterations)')
plt.ylabel('Test F1 Score')

# Add a title
plt.title('Test F1 Score vs Epoch for Different Dimensionality Reduction Techniques')

# Show legend
plt.legend(loc='best')

# Add grid for clarity
plt.grid(True)

# Display the plot
plt.show()











################ Step 5: Neural Network with Cluster Features ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the KMeans model with the chosen number of clusters on the training set (e.g., k = 2)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)

# Step 2: Append K-Means clusters as a new feature to X_train and X_test
X_train_kmeans = np.hstack((X_train, kmeans.labels_.reshape(-1, 1)))  # Append K-means cluster labels to X_train
X_test_kmeans = np.hstack((X_test, kmeans.predict(X_test).reshape(-1, 1)))  # Append K-means cluster labels to X_test

# Step 3: Define the parameter grid for grid search (for MLPClassifier)
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Step 4: Create the pipeline with StandardScaler and MLPClassifier with your specific hyperparameters
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed,
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Step 5: Set up GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Step 6: Fit the grid search on the new training set with appended K-Means clusters
grid_search.fit(X_train_kmeans, Y_train)

# Step 7: Print the best parameters and corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1 score (weighted): {grid_search.best_score_:.4f}")

# Step 8: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
Y_test_pred = best_model.predict(X_test_kmeans)
test_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')
print(f"Test F1 Score (weighted): {test_f1_score:.4f}")

############## Best model Kmeans NN ####################
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, KFold

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define KMeans clustering and append cluster labels as features
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)
X_train_kmeans = np.hstack((X_train, kmeans.labels_.reshape(-1, 1)))  # Append K-means cluster labels to X_train
X_test_kmeans = np.hstack((X_test, kmeans.predict(X_test).reshape(-1, 1)))  # Append K-means cluster labels to X_test

# Define the MLPClassifier with fixed configuration
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam',
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline_mlp = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features
    ('mlp', mlp_model)  # Neural network classifier with the fixed configuration
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(pipeline_mlp, X_train_kmeans, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction_mlp = round(end_time - start_time, 4)

# Train the model on the full training set with KMeans features
start_time = time.time()
pipeline_mlp.fit(X_train_kmeans, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = pipeline_mlp.predict(X_test_kmeans)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations_mlp = pipeline_mlp.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted_mlp = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted_mlp = f1_score(Y_test, Y_test_pred_mlp, average='weighted')

# Print out timing, convergence, and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction_mlp} seconds")
print(f"Training Time for the MLP model: {time_mlp_training} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Number of Iterations for MLP Convergence: {mlp_iterations_mlp}")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted_mlp:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted_mlp:.4f}")

############## Best model Kmeans NN ####################
# Cross-validation Prediction Time: 4.4123 seconds
# Training Time for the MLP model: 0.9881 seconds
# Prediction Time for the MLP model: 0.0055 seconds
# Number of Iterations for MLP Convergence: 217
# Cross-validated Training F1 Score (Weighted): 0.8074
# Test F1 Score (Weighted): 0.8245


############ EM NN Grid Search ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the GMM model with the chosen number of components on the training set (e.g., n_components = 2)
optimal_component = 2
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)

# Step 2: Append GMM cluster predictions as a new feature to X_train and X_test
X_train_gmm = np.hstack((X_train, gmm.predict(X_train).reshape(-1, 1)))  # Append GMM labels to X_train
X_test_gmm = np.hstack((X_test, gmm.predict(X_test).reshape(-1, 1)))     # Append GMM labels to X_test

# Step 3: Define the parameter grid for grid search (for MLPClassifier)
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Step 4: Create the pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed,
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Step 5: Set up GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Step 6: Fit the grid search on the new training set with appended GMM components
grid_search.fit(X_train_gmm, Y_train)

# Step 7: Print the best parameters and corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1 score (weighted): {grid_search.best_score_:.4f}")

# Step 8: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
Y_test_pred = best_model.predict(X_test_gmm)
test_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')
print(f"Test F1 Score (weighted): {test_f1_score:.4f}")

########## Best Model EM NN ############
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the GMM model with the chosen number of components on the training set (e.g., n_components = 2)
optimal_component = 2
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)

# Step 2: Append GMM cluster predictions as a new feature to X_train and X_test
X_train_gmm = np.hstack((X_train, gmm.predict(X_train).reshape(-1, 1)))  # Append GMM labels to X_train
X_test_gmm = np.hstack((X_test, gmm.predict(X_test).reshape(-1, 1)))     # Append GMM labels to X_test

# Step 3: Define the MLPClassifier with a fixed hidden layer configuration (32, 16)
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam',
                          random_state=seed, batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features
    ('mlp', mlp_model)  # Neural network classifier with (32, 16) hidden layers
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(pipeline, X_train_gmm, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Train the model on the full training set with GMM features
start_time = time.time()
pipeline.fit(X_train_gmm, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = pipeline.predict(X_test_gmm)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations = pipeline.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted = f1_score(Y_test, Y_test_pred_mlp, average='weighted')

# Print out timing, convergence, and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Training Time for the MLP model: {time_mlp_training} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Number of Iterations for MLP Convergence: {mlp_iterations}")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

############### EM Neural Network Model Learning Curve ############
# Cross-validation Prediction Time: 7.9001 seconds
# Training Time for the MLP model: 1.1539 seconds
# Prediction Time for the MLP model: 0.0057 seconds
# Number of Iterations for MLP Convergence: 248
# Cross-validated Training F1 Score (Weighted): 0.8142
# Test F1 Score (Weighted): 0.8322

################## K-Means Neural Network with Cluster Features ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the KMeans model with the chosen number of clusters on the training set (e.g., k = 2)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)

# Step 2: Only use KMeans cluster labels as features (ignore original features)
X_train_kmeans_only = kmeans.labels_.reshape(-1, 1)  # Use only the KMeans labels for X_train
X_test_kmeans_only = kmeans.predict(X_test).reshape(-1, 1)  # Use only the KMeans labels for X_test

# Step 3: Define the parameter grid for grid search (for MLPClassifier)
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Step 4: Create the pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features (though KMeans labels may not need it)
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed,
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Step 5: Set up GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Step 6: Fit the grid search on the KMeans cluster labels (no original features)
grid_search.fit(X_train_kmeans_only, Y_train)

# Step 7: Print the best parameters and corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1 score (weighted): {grid_search.best_score_:.4f}")

# Step 8: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
Y_test_pred = best_model.predict(X_test_kmeans_only)
test_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')
print(f"Test F1 Score (weighted): {test_f1_score:.4f}")

# Optional: You can also measure times similar to the previous approach (cross-validation, training, prediction time)
# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(grid_search.best_estimator_, X_train_kmeans_only, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = best_model.predict(X_test_kmeans_only)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Print out timing and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")

############## Best Model Kmeans Only NN ####################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the KMeans model with the chosen number of clusters on the training set (e.g., k = 2)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=seed)
kmeans.fit(X_train)

# Step 2: Only use KMeans cluster labels as features (ignore original features)
X_train_kmeans_only = kmeans.labels_.reshape(-1, 1)  # Use only the KMeans labels for X_train
X_test_kmeans_only = kmeans.predict(X_test).reshape(-1, 1)  # Use only the KMeans labels for X_test

# Step 3: Define the MLPClassifier with the best performing hidden layer configuration (16,)
mlp_model = MLPClassifier(hidden_layer_sizes=(16,), activation='tanh', solver='adam',
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features (though KMeans labels may not need it)
    ('mlp', mlp_model)  # Neural network classifier with (16,) hidden layers
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(pipeline, X_train_kmeans_only, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Train the model on the full training set with KMeans features
start_time = time.time()
pipeline.fit(X_train_kmeans_only, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = pipeline.predict(X_test_kmeans_only)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations = pipeline.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted = f1_score(Y_test, Y_test_pred_mlp, average='weighted')

# Print out timing, convergence, and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Training Time for the MLP model: {time_mlp_training} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Number of Iterations for MLP Convergence: {mlp_iterations}")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

############## Best Model Kmeans Only NN ####################
# Cross-validation Prediction Time: 8.0882 seconds
# Training Time for the MLP model: 3.6659 seconds
# Prediction Time for the MLP model: 0.023 seconds
# Number of Iterations for MLP Convergence: 304
# Cross-validated Training F1 Score (Weighted): 0.7353
# Test F1 Score (Weighted): 0.7243

################## EM Only Neural Network with Cluster Features ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the GMM model with the chosen number of components on the training set (e.g., n_components = 2)
optimal_component = 2
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)

# Step 2: Only use GMM cluster predictions as features (ignore original features)
X_train_gmm_only = gmm.predict(X_train).reshape(-1, 1)  # Use only the GMM labels for X_train
X_test_gmm_only = gmm.predict(X_test).reshape(-1, 1)    # Use only the GMM labels for X_test

# Step 3: Define the parameter grid for grid search (for MLPClassifier)
param_grid = {
    'mlp__hidden_layer_sizes': [
        (16,),           # 1 hidden layer with 16 units
        (32,),           # 1 hidden layer with 32 units
        (64,),           # 1 hidden layer with 64 units
        (32, 16),        # 2 hidden layers with 32 and 16 units
        (64, 32),        # 2 hidden layers with 64 and 32 units
        (64, 32, 16),    # 3 hidden layers with 64, 32, and 16 units
    ]
}

# Step 4: Create the pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features (though GMM labels may not need it)
    ('mlp', MLPClassifier(activation='tanh', solver='adam', random_state=seed,
                          batch_size=128, learning_rate_init=0.0001, alpha=0.001, max_iter=500))
])

# Step 5: Set up GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Step 6: Fit the grid search on the GMM cluster labels (no original features)
grid_search.fit(X_train_gmm_only, Y_train)

# Step 7: Print the best parameters and corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1 score (weighted): {grid_search.best_score_:.4f}")

# Step 8: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
Y_test_pred = best_model.predict(X_test_gmm_only)
test_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')
print(f"Test F1 Score (weighted): {test_f1_score:.4f}")

# Optional: You can also measure times similar to the previous approach (cross-validation, training, prediction time)
# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(grid_search.best_estimator_, X_train_gmm_only, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = best_model.predict(X_test_gmm_only)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Print out timing and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")

# Best parameters found: {'mlp__hidden_layer_sizes': (32, 16)}

############# Best Model EM Only NN ################
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Step 1: Fit the GMM model with the chosen number of components on the training set (e.g., n_components = 2)
optimal_component = 2
gmm = GaussianMixture(n_components=optimal_component, covariance_type='full', random_state=seed)
gmm.fit(X_train)

# Step 2: Only use GMM cluster predictions as features (ignore original features)
X_train_gmm_only = gmm.predict(X_train).reshape(-1, 1)  # Use only the GMM labels for X_train
X_test_gmm_only = gmm.predict(X_test).reshape(-1, 1)    # Use only the GMM labels for X_test

# Step 3: Define the MLPClassifier with the best performing hidden layer configuration (32, 16)
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam',
                          random_state=seed, batch_size=128, learning_rate_init=0.0001,
                          alpha=0.001, max_iter=500)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features (though GMM labels may not need it)
    ('mlp', mlp_model)  # Neural network classifier with (32, 16) hidden layers
])

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Measure wall clock time for cross-validation predictions on the training data
start_time = time.time()
Y_train_pred_mlp = cross_val_predict(pipeline, X_train_gmm_only, Y_train, cv=kf)
end_time = time.time()
time_cv_prediction = round(end_time - start_time, 4)

# Train the model on the full training set with GMM features
start_time = time.time()
pipeline.fit(X_train_gmm_only, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for making predictions on the test set
start_time = time.time()
Y_test_pred_mlp = pipeline.predict(X_test_gmm_only)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations = pipeline.named_steps['mlp'].n_iter_

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred_mlp, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted = f1_score(Y_test, Y_test_pred_mlp, average='weighted')

# Print out timing, convergence, and performance metrics
print(f"Cross-validation Prediction Time: {time_cv_prediction} seconds")
print(f"Training Time for the MLP model: {time_mlp_training} seconds")
print(f"Prediction Time for the MLP model: {time_mlp_prediction} seconds")
print(f"Number of Iterations for MLP Convergence: {mlp_iterations}")
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

# Cross-validation Prediction Time: 0.9409 seconds
# Training Time for the MLP model: 0.2166 seconds
# Prediction Time for the MLP model: 0.0068 seconds
# Number of Iterations for MLP Convergence: 56
# Cross-validated Training F1 Score (Weighted): 0.6455
# Test F1 Score (Weighted): 0.6791