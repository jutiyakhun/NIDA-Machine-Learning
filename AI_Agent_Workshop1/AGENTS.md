# Procedure
Analyst data from https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

## 1. Explore Data
**1.1.** Ignore unnecessary column e.g. index column

**1.2.** Assign type of data
* **a. Numerical variables**
    * i. Fill missing value using mode value
    * ii. Encode them using one-hot encoder
* **b. Categorical variables**
    * i. Fill missing value using median value

**1.3.** Visualization
* **a. Numerical variables**
    * i. Visualizes relationships between each quantity variable and "CUST_ID" within a single chart
    * ii. Visualize box plots of each quantity variable 
* **b. Categorical variables**
    * i. Visualize bar chart to demonstrate frequency of "CUST_ID" in each qualitative variable

**1.4.** Normalize all Numerical variables and combine them with Categorical variables in “NORM_DATA”

## 2. Dimension Reduction
**2.1.** Use PCA in “NORM_DATA” and return output in “PCA_DATA”

## 3. Clustering
**3.1.** Use K-Means to cluster the “PCA_DATA”

**3.2.** Use DBSCAN to cluster the “PCA_DATA”, and tunning it

## 4. Select the best number of groups (k) in K-Means; K is in [2,10]
**4.1.** Use Elbow Method to do

**4.2.** Plot Silhouette scores 

## 5. Demonstrate the best K and Silhouette scores of K-Means, and DBSCAN

## 6. Show scatter plots of data classified by group