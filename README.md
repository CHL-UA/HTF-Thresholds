Establishing flood thresholds for sea level rise impact communication

## Summary
In this work, we have devised a methodology to estimate minor coastal flooding (high tide flooding, HTF) thresholds at ungauged basins along the US coastlines (West, Gulf, and Atlantic coastlines). To achieve this goal, we utilized a machine learning algorithm, specifically random forest regression, due to its ability to capture nonlinear interactions between influential components affecting HTF thresholds. However, these influential factors exhibit diverse characteristics across different regions, making it challenging for a single random forest regressor to accurately capture this variability.

To address this issue, we employed K-means clustering to group similar influential components (input features to random forest) into clusters. Subsequently, we developed distinct random forest models with different parameters for each cluster, thereby enhancing the model's ability to capture regional variability.

Furthermore, ensuring the availability of influential components at every ungauged basin was crucial for predicting and estimating HTF thresholds. Among these components, sea level rise (SLR) posed a challenge as it is not readily available at ungauged basins. To overcome this limitation, we implemented random forest regression to predict SLR rates (as the target variable) using influential factors on SLR as input features.

In the final step, we trained and validated the random forest regressor to estimate HTF thresholds at ungauged basins, leveraging the SLR rates obtained from the previous step. This comprehensive approach allowed us to provide accurate estimates of minor coastal flooding thresholds, contributing to better coastal hazard assessment and management.

## Details
This repository hosts Python scripts for 1) cluster the input features of HTF threhsolds random forest into different classifications (**clustering_kmeans.py**); 2) estimating SLR rates at ungauged basins (**Code_for_estimating_SLR_rates_RF.py**); 2) estimating HTF thresholds at ungauged basins (**Code_for_estimating_HTF_thresholds_RF.py**). All the input and output data have been provided in each folder.