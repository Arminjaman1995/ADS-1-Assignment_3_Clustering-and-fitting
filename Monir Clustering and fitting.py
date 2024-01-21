#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd

# Load the dataset
file_path = 'emission data.csv'
data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure
data.head()


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Focusing on recent years (e.g., last 40 years)
recent_years = data.columns[-40:]

# Extracting the recent data
recent_data = data[recent_years]

# Normalizing the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(recent_data)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=0)  # Using 5 clusters as an example
clusters = kmeans.fit_predict(normalized_data)

# Adding the cluster information to the original dataframe
data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(15, 10))
for i in range(kmeans.n_clusters):
    # Plotting each cluster
    plt.scatter(recent_data[clusters == i][recent_years[-1]], recent_data[clusters == i][recent_years[-2]], label=f'Cluster {i}')

# Marking the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, -1], centers[:, -2], s=300, c='black', marker='x', label='Centers')

plt.xlabel('Emissions in ' + recent_years[-1])
plt.ylabel('Emissions in ' + recent_years[-2])
plt.title('CO2 Emissions Clustering (Last two years)')
plt.legend()
plt.show()


# In[3]:


# Displaying a snippet of the data with cluster labels
data[['Country', 'Cluster'] + list(recent_years)].head()


# In[4]:


from scipy.optimize import curve_fit

# Choosing a cluster for analysis (e.g., Cluster 0)
cluster_data = data[data['Cluster'] == 0][recent_years]

# Example: Modeling CO2 emissions for the last 40 years using an exponential growth model
# Define an exponential growth model
def exp_growth(x, a, b, c):
    return a * np.exp(b * x) + c

# Prepare data for curve fitting
years = np.arange(0, len(recent_years))  # Using relative years for fitting
emissions = cluster_data.mean()  # Using mean emissions of the cluster

# Curve fitting
params, params_covariance = curve_fit(exp_growth, years, emissions, p0=[1, 0.01, 1])

# Making predictions for the next 20 years
future_years = np.arange(len(recent_years), len(recent_years) + 20)
predicted_emissions = exp_growth(future_years, *params)

# Plotting the results
plt.figure(figsize=(15, 10))
plt.scatter(years, emissions, label='Observed Emissions')
plt.plot(future_years, predicted_emissions, color='red', label='Predicted Emissions')
plt.xlabel('Years since 1978')
plt.ylabel('CO2 Emissions')
plt.title('Exponential Growth Model of CO2 Emissions for Cluster 0')
plt.legend()
plt.show()

# Parameters of the fitted model
params


# In[5]:


# To improve the model and gain more insights, we can include confidence ranges for the predictions.
# The 'err_ranges' function can be used for this purpose. Since it's not provided, we'll create a similar function.

def err_ranges(x, popt, pcov):
    """
    Estimate the error ranges for the fitted curve.
    
    :param x: The x-values at which to evaluate the error.
    :param popt: Optimal values for the parameters.
    :param pcov: The estimated covariance of popt.
    :return: lower and upper limits of the confidence range.
    """
    if pcov is None:
        # Can't calculate confidence range without covariance matrix
        return None, None
    
    perr = np.sqrt(np.diag(pcov))
    predictions = exp_growth(x, *popt)
    lower = exp_growth(x, *(popt - perr))
    upper = exp_growth(x, *(popt + perr))

    return lower, upper

# Calculating the confidence ranges for future predictions
lower_bounds, upper_bounds = err_ranges(future_years, params, params_covariance)

# Plotting the results with confidence intervals
plt.figure(figsize=(15, 10))
plt.scatter(years, emissions, label='Observed Emissions')
plt.plot(future_years, predicted_emissions, color='red', label='Predicted Emissions')
plt.fill_between(future_years, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Confidence Interval')
plt.xlabel('Years since 1978')
plt.ylabel('CO2 Emissions')
plt.title('Exponential Growth Model of CO2 Emissions for Cluster 0 with Confidence Ranges')
plt.legend()
plt.show()


# In[ ]:




