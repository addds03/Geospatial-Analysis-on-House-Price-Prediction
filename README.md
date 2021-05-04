# Toronto House Prices: A Hedonic Geospatial Analysis: Overview

* Created a tool that estimates house prices (RMSLE ~ 0.1325) to help house buyers, sellers, or agent negotiate their house prices.
* Gathered data from multiple data sources, primary zoocasa.ca, and external factors Toronto Open Data Portal, Airbnb.ca, Toronto Police Service Data Portal. 
* Engineered 15 features from 8 different datasets using Python.
*	Implemented geospatial proximity joins using K-dimensional tree package from 'scipy' to engineer features on shapefile.
* Used Hedonic Spatial Regression to determine elastic 10% increase of features on the final price.
* Designed a stacked meta-model combining ENET, GBDT, and KRR models, with linear regression as our base model to get 85% accuracy.
* Devised a final model using weighted averages 70%-15%-15% for the stacked model, LightGBM, and XGBoost respectively to reach 87% accuracy.


## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, scipy
**Visualizations:** Tableau, matplotlib, seaborn, brewer, folium


## Data Cleaning and Preprocessing

After collecting the data, we needed to clean it up so that it was usable for our model. we made the following changes and created the following variables:

*	The dataset is missing a significant amount is sqft data (30%), Imputed using Random forest.
*	Real estate features extracted from 'zoocasa' dataset - Except neighborhoods all the other features we easily available. They were assigned to the house using geospatial merge; a geojson file that combined polygons outlining neighborhood boundaries.

| ID |   Lat,Long   |  Sqft  | Parking | Neighborhood | House Type | Bedrooms | Bathrooms |
|:--:|:------------:|:------:|:-------:|:-------------:|:----------:|:--------:|:---------:|
| 21 | 43.66,-79.38 | 1050.0 |    2    |     Niagra    |  Detached  |     3    |     2     |
| 22 | 43.67,-79.53 |  805.0 |    1    |    Malvern    |  Apartment |     2    |     1     |

* Extracted features in the final dataset - These features excluding Median Income, are obtained using Geospatial proximity joins. To count the number of thefts, Violent crimes, and Airbnb's within 500metres of a house, we used a technique considering the house as the center of a circle (500m radius) for each row. For Airbnb, we take count of the year 2014 and 2019 to obtain YOY growth change. In contrast, we calculated the sum of all violent and auto thefts from 2014-2019. We calculate the euclidean distance to features engineer the distances. We extracted median income using the neighborhood profiles dataset and joined it to the housing dataset.

| ID | Theft500m | Violent500m | Airbnb500m | Median Income | Distance to Hospital | Distance to Parks | Distance to Transit stops | Distance to School |
|:--:|:---------:|:-----------:|:----------:|:-------------:|:--------------------:|:-----------------:|:-------------------------:|:------------------:|
| 21 |     45    |      18     |     14     |     85526     |        130.27        |       129.47      |           54.32           |        61.6        |
| 22 |     70    |      24     |     20     |     56700     |        111.67        |       150.7       |            63.1           |        88.9        |

## Exploratory Data Analysis

