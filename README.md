# Toronto House Prices: A Hedonic Geospatial Analysis: Overview (Aditya Gaikwad & Jacob Hazen)

* Created a tool that estimates house prices (RMSLE ~ 0.1325) to help house buyers, sellers, or agent negotiate their house prices.
* Gathered data from multiple data sources, primary Zoocasa.ca, and external factors Toronto Open Data Portal, Airbnb.ca, Toronto Police Service Data Portal. 
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

* Missing sqft data was mainly from the detached and semi-detached houses.

![Screenshot 2021-05-04 at 5 35 54 PM](https://user-images.githubusercontent.com/39771193/117072912-302fa980-acff-11eb-9666-2eb68e778c07.png)

* Spatial heatmap of median income, shows higher income (red) in the center (downtown).

![Screenshot 2021-05-04 at 5 40 35 PM](https://user-images.githubusercontent.com/39771193/117073400-d7144580-acff-11eb-9b6f-9e09a3409b2b.png)

* Crimes in Toronto, 2014 to 2019

| Assualt | Breaking & Entering | Auto Theft | Robbery | Theft over $1000 |
|:-------:|:-------------------:|:----------:|:-------:|:----------------:|
|  51.5%  |        23.7%        |    11.8%   |   9.4%  |       3.7%       |

*P.S - Homicide were excluded from the dataset due to being sensitive information.*

* Elastice 10% increase of features on final price

|           Features           | Percentage Impact | $Amount/Million |
|:----------------------------:|:-----------------:|:---------------:|
|         Median Income        |       9.46%       |      $94600     |
|          Square Feet         |       4.40%       |      $44000     |
|      Distance to School      |       0.07%       |       $700      |
|   Distance to Transit Stop   |       -0.10%      |      -$100      |
|       Auto Thefts 500m       |       -0.02%      |      -$200      |
|        Violent Crimes        |       -0.23%      |      -$2300     |
|       Distance to Parks      |       -0.04%      |      -$400      |
| Distance to Nearest Hospital |       0.02%       |       $200      |

* Semi-elastic 1 unit increase of feature on final price

| Features | Percentage Impact | $Amount/Million |
|:--------:|:-----------------:|:---------------:|
|  Parking |       2.63%       |      $22630     |
| Bathroom |       6.57%       |      $65700     |


## Methodology

### Casual Analysis

To build a good regression model, we check
1. Missing values
2. Multicolinearity - Using pearson corelation, only one of the features with coeffiecient greater than 0.9 were used.
3. Outliers - By performing scatter plot.
4. Normality - Variables were right-skewed, we performed logarithmic transformation

* Hedonic price equation to measure house cost:

![Screenshot 2021-05-04 at 6 05 00 PM](https://user-images.githubusercontent.com/39771193/117075684-40e21e80-ad03-11eb-98f9-1462448adf3a.png)

where the price is the final sale price of the house, ACrime is the amount of auto theft-related crimes within 500 meters of the house, Vcrime is the number of violent crimes within 500 meters of the house, ΔAirbnb is the change of Airbnb’s within 500 meters of the house from 2014 to 2019, distPark is the measured euclidean distance from the nearest park to the house, distHosp is the measured euclidean distance from the nearest hospital to the house. Both distPark and distHosp are measured in meters.
Additionally, we included dummy variables for both house type and neighborhood to capture the individual house types and unobserved neighborhood heterogeneities.

### Machine Learning

* We performed preprocessing, mentioned in the earlier sections.
* 80-20 Train-Test split.
* To cross-validate our model, we chose a stratified 5-folds.
* We choose Root mean square log error (RMSLE), because it only considers the relatie error, and also includes larger penalty for underestimation.

![Screenshot 2021-05-04 at 6 14 10 PM](https://user-images.githubusercontent.com/39771193/117076528-881cdf00-ad04-11eb-9031-626a51d24000.png)

### Model Performance

![Screenshot 2021-05-04 at 6 15 03 PM](https://user-images.githubusercontent.com/39771193/117076618-a7b40780-ad04-11eb-810b-20fc79324d16.png)

***We created a stacked meta-model combining ENET, GBDT, and KRR models, with linear regression as our base model. These models were selected, as we wanted to avoid the overfitting characteristic of tree-based models while improving the performance of our under-performing algorithms by combining them. The stacked meta-model performed well, with an RMSLE of 0.15 on the training data and 0.1717 on the testing data. To further improve the performance of our stacked model, we combined our stacked, LightGBM, and XGBoost models with weighted averages. The stacked model was assigned a more prominent importance of 70%, followed by 15% each for the LightGBM and XGBoost models. By doing this, we were able to improve our performance from an RMSLE of 0.1356 to 0.1325.***
