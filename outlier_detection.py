import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load the CSV file
file_path = 'AKWA IBOM_crosschecked.csv'
AK = pd.read_csv(file_path)

# Data preprocessing
AK['id'] = AK.index
Ak = pd.DataFrame({"id": AK['id'], "Latitude": AK['Latitude'], "Longitude": AK['Longitude']})

# Identifying neighbouring polling units
def get_neighbours(polling_units_df, radius_km=1):
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        polling_units_df,
        geometry=gpd.points_from_xy(polling_units_df['Longitude'], polling_units_df['Latitude']),
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    )
    
    # Create a spatial index
    gdf_sindex = gdf.sindex

    neighbours = []

    for idx, unit in gdf.iterrows():
        # Get the bounding box for the search area
        point = unit['geometry']
        possible_matches_index = list(gdf_sindex.intersection(point.buffer(radius_km / 111.32).bounds))  # Convert km to degrees
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches['geometry'].distance(point) <= radius_km / 111.32]
        
        for _, other_unit in precise_matches.iterrows():
            if unit['id'] != other_unit['id']:  # Ensure not comparing with itself
                neighbours.append((unit['id'], other_unit['id']))  # 'id' is present
    
    return neighbours


# Call the function with the DataFrame
neighbours = get_neighbours(Ak, radius_km=1)

# Example output: Print the first 5 neighbours for inspection
print(neighbours[:5])

# Convert the list of neighbours to a DataFrame
neighbours_df = pd.DataFrame(neighbours, columns=['Unit_ID', 'Neighbour_ID'])
neighbours_df.head()

# data cleaning for `neighbours_df` dataframe
neighbours_df[neighbours_df['Unit_ID'] == 1001]

neighbours_df_dropped = neighbours_df[:21523]
neighbours_df_dropped[neighbours_df_dropped['Unit_ID'] == 1001]

# loading dataset
file_path = 'AKWA IBOM_crosschecked.csv'
akwa = pd.read_csv(file_path)

#data prepping
akwa.drop('Location', axis=1, inplace=True)
akwa.head(1)

ndf =  neighbours_df
ndf = ndf.sort_values(by='Unit_ID', ascending=True)

# Group the DataFrame by 'Unit_ID'
grouped_ndf = ndf.groupby('Unit_ID')

# Aggregate 'Neighbour_ID' into a list using the 'list' aggregation function
transformed_ndf = grouped_ndf['Neighbour_ID'].agg(list)

# Reset the index to obtain a regular DataFrame with 'Unit_ID' as a column
transformed_ndf = transformed_ndf.reset_index()

print(transformed_ndf)

# Function to calculate outlier scores
def calculate_outlier_score(df, neighbour_df):
    outlier_scores = []
    outliers = []
    threshold = 3

    for index, unit in df.iterrows():
        unit_id = unit['id']
        
        # Find neighbors for the current unit
        neighbours = neighbour_df[neighbour_df['Unit_ID'] == unit_id]['Neighbour_ID']
        
        if neighbours.empty:
            continue
        
        for party in ['APC', 'LP', 'PDP', 'NNPP']:  # replace with actual party columns
            # Get votes for the current party from all neighboring units
            neighbour_votes = df[df['id'].isin(neighbours)][party]
            
            if not neighbour_votes.empty:
                avg_neighbor_votes = np.mean(neighbour_votes)
                std = np.std(neighbour_votes)
                
                # Skip calculations if the standard deviation is zero
                if std == 0:
                    continue
                
                # Calculate z-scores for the current unit
                unit_votes = unit[party]
                unit_z_score = (unit_votes - avg_neighbor_votes) / std
                outlier_scores.append({
                    'id': unit_id,
                    'Party': party,
                    'Outlier_Score': unit_z_score
                })

                # Calculate z-scores for neighbors and check against the threshold
                for votes in neighbour_votes:
                    z_score = (votes - avg_neighbor_votes) / std
                    if np.abs(z_score) > threshold:
                        outliers.append({
                            'id': unit_id,
                            'Neighbour_ID': neighbours.tolist(),  # Convert to list
                            'Party': party,
                            'Votes': votes,
                            'Z-Score': z_score
                        })
    
    return pd.DataFrame(outlier_scores), pd.DataFrame(outliers)

# Calculate outlier scores and outliers
outlier_scores_df, outliers_df = calculate_outlier_score(AK, neighbours_df)

outlier_scores_df

# Merge outlier scores with the original dataframe
AK = AK.merge(outlier_scores_df, on='id', how='left')

# Display the results
print(AK)
print(outliers_df)

# Sort by outlier score
sorted_outliers = AK.sort_values(by='Outlier_Score', ascending=False)

# Save the cleaned dataset with outlier scores
sorted_outliers.to_csv('cleaned_dataset_with_outliers.csv', index=False)
sorted_outliers

sorted_outliers_df.to_csv('cleaned_dataset_with_outlier_votes.csv', index=False)

# Visualizations

# Histogram of outlier scores
plt.figure(figsize=(10, 6))
sns.histplot(sorted_outliers['Outlier_Score'].dropna(), bins=20, kde=True)
plt.title('Distribution of Outlier Scores')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')

# Scatter plot of outlier scores geographically
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sorted_outliers, x='Longitude', y='Latitude', hue='Outlier_Score', palette='coolwarm', size='Outlier_Score', sizes=(20, 200), legend=False)
plt.title('Geographical Visualization of Outlier Scores')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Find the unit ID with the highest outlier score
unit_with_highest_score = sorted_outliers[sorted_outliers['Outlier_Score'] == sorted_outliers['Outlier_Score'].max()]['Unit_id'].values[0]



# Get neighboring unit IDs for the unit with the highest score
# Get the lists of Neighbour_IDs
neighbour_ids_lists = sorted_outliers[sorted_outliers['Unit_id'] == unit_with_highest_score]['Neighbour_ID'].to_list()

# Flatten the list and remove duplicates
neighbouring_unit_ids = list(set([item for sublist in neighbour_ids_lists for item in sublist]))

# Filter data for plotting (assuming 'Longitude' and 'Latitude' columns for location)
filtered_df = sorted_outliers[sorted_outliers['Unit_id'].isin([unit_with_highest_score] + neighbouring_unit_ids)]

# Extract data for plotting
unit_x = filtered_df[filtered_df['Unit_id'] == unit_with_highest_score]['Longitude'].values[0]
unit_y = filtered_df[filtered_df['Unit_id'] == unit_with_highest_score]['Latitude'].values[0]
neighbor_xs = filtered_df[filtered_df['Unit_id'].isin(neighbouring_unit_ids)]['Longitude'].tolist()
neighbor_ys = filtered_df[filtered_df['Unit_id'].isin(neighbouring_unit_ids)]['Latitude'].tolist()

unit_x,unit_y

# Create the scatter plot (assuming you have matplotlib.pyplot imported as plt)
plt.figure(figsize=(10, 6))
plt.scatter(unit_x, unit_y, s = 50, color='red', label='Unit with Highest Score')
plt.scatter(neighbor_xs, neighbor_ys, s = 20, color='blue', alpha=0.7, label='Neighboring Units')

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Unit with Highest Outlier Score and Neighbors')
plt.legend()
plt.show()





def plot_party_votes(sorted_outliers, outlier_score):
    # Find the unit ID with the given outlier score
    unit_with_highest_score = sorted_outliers[sorted_outliers['Outlier_Score'] == outlier_score]['Unit_id'].values[0]
    print("Unit with the highest score:", unit_with_highest_score)
    
    # Get neighboring unit IDs for the unit with the given outlier score
    neighbour_ids_lists = sorted_outliers[sorted_outliers['Unit_id'] == unit_with_highest_score]['Neighbour_ID'].to_list()
    neighbouring_unit_ids = list(set([item for sublist in neighbour_ids_lists for item in sublist]))
    print("Neighboring unit IDs:", neighbouring_unit_ids)

    # Filter data for the unit and its neighbors
    filtered_df = sorted_outliers[sorted_outliers['Unit_id'].isin([unit_with_highest_score] + neighbouring_unit_ids)]
    print("Filtered DataFrame:\n", filtered_df)

    # Extract votes for each party
    parties = ['APC', 'PDP', 'LP', 'NNPP']
    units = filtered_df['Unit_id'].unique().tolist()
    print("Polling Units:", units)
    
    votes_data = {party: [filtered_df[(filtered_df['Unit_id'] == unit) & (filtered_df['Party'] == party)][party].sum() for unit in units] for party in parties}
    print("Votes Data:", votes_data)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    bar_width = 0.2
    indices = range(len(units))

    # Plot bars for each party
    for i, party in enumerate(parties):
        ax.bar([index + i * bar_width for index in indices], votes_data[party], bar_width, label=party)
    
    # Set labels and title
    ax.set_xlabel('Polling Units')
    ax.set_ylabel('Votes')
    ax.set_title('Votes for Parties in Unit(2nd highest Outlier Score) and Neighboring Units')
    ax.set_xticks([index + bar_width * (len(parties) - 1) / 2 for index in indices])
    ax.set_xticklabels(units, rotation=45)
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# Example usage
plot_party_votes(sorted_outliers, outlier_score=120.20815280171307)


plt.show()



