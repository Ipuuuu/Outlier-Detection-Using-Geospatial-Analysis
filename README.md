# Outlier-Detection-Using-Geospatial-Analysis
 This project focuses on identifying outlier polling units in Akwa-Ibom State, Nigeria based on the votes each party received.
 The analysis will involve geospatial techniques to find neighbouring polling units and calculate an outlier score for each party
 in each unit. The goal is to pinpoint polling units where the voting results significantly deviate from their neighbours,
 indicating potential irregularities or influences.

 The file `AKWA IBOM_crosschecked ` contains the data required on the state's polling units. It includes the longititude and latitudes
 of polling units obtained using geocoding techniques. Steps taken in this project include:
  ## Dataset Preparation:
  - Open the file `AKWA IBOM_crosschecked ` and download it
    
  ## Neighbour Identification:
  - Identify neighbouring polling units based on geographical proximity. Define a radius (e.g., 1 km) to determine which units are considered neighbours.
    
  ## Outlier Score Calculation:
  - For each polling unit, compare the votes each party received with those of its neighbouring units.
  - Calculate an outlier score for each party based on the deviation of votes from neighbouring units.
  - Record the outlier scores along with the respective parties and neighbouring units.
    
  ## Sorting and Reporting:
  - Sort the dataset by the outlier scores for each party to identify the most significant outliers.
  - Provide a detailed report explaining the methodology and findings.
  - Highlight the top 3 outliers and their closest polling units, explaining why they are considered outliers.

### A detailed report containing:
- Explanation of the methodology used for geospatial analysis and outlier detection.
- Summary of findings, including the sorted list of polling units by outlier scores for each party.
- Detailed examples of the top 3 outliers, with explanations and visualizations.
- A conclusion summarizing the findings, including key insights and visualizations.
Can be found in the file `OUTLIER DETECTION`

### A cleaned and sorted CSV file showing a sorted list of polling units by outlier scores for each party.
Can be found in the file `cleaned_dataset_with_outliers`



