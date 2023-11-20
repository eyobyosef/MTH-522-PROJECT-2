import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the Excel file 1 into a pandas DataFrame
data = pd.read_excel(r'C:\Users\meggn\Documents\MTH 522\Project 2\fatalpolice-shootings-data.xlsx')
# Load the Excel file 2 into a pandas DataFrame
df = pd.read_excel(r'C:\Users\meggn\Documents\MTH 522\Project 2\fatalpolice-shootings-data-web.xlsx')
print(data.head()) 
print(data.info()) 
print(data.describe()) 
print(df.head()) 
print(df.info()) 
print(df.describe()) 
# Q. Are there any demographic disparities in police shootings?
# Create a histogram of the ages of individuals involved in shootings
import matplotlib.pyplot as plt
import seaborn as sns 
plt.figure(figsize=(10, 5))
sns.histplot(data['age'], kde=True, bins=25)
plt.title('Age Distribution in Police Shootings')
plt.xlabel('Age')
plt.ylabel('Number of Incidents')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
# Drop rows with missing values in the 'age' column
data = data.dropna(subset=['age'])
# Find the bin with the maximum shootings
max_shootings_bin = counts.argmax()
# Calculate the age range for the bin with the maximum shootings
age_range_start = int(bin_edges[max_shootings_bin])
age_range_end = int(bin_edges[max_shootings_bin + 1])
print("Q. Calculate the age range for the bin with the maximum 
shootings.")
print(f"The age range with the maximum number of shootings is from 
{age_range_start} to {age_range_end} years.")
# Q. write a code for age distribution across race
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Filter the data for each race
races = data['race'].unique()
# Create a separate histogram for each race
plt.figure(figsize=(12, 8))
for race in races:
 sns.histplot(data=data[data['race'] == race], x='age', label=race, 
kde=True, alpha=0.6)
plt.title('Age Distribution Across Race')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# Q. Which race is the most killed one?
import matplotlib.pyplot as plt
import seaborn as sns
# Create a bar plot to show the distribution of shootings by race
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='race', 
order=data['race'].value_counts().index)
plt.title('Distribution of Shootings by Race')
plt.xlabel('Race')
plt.ylabel('Number of Shootings')
plt.xticks(rotation=45)
plt.show()
print("Q. Which race is the most killed one?")
most_killed_race = data['race'].value_counts().idxmax()
print(f" The most killed race is: {most_killed_race}")
white_ages = df[df['race'] == 'W']['age'].dropna()
black_ages = df[df['race'] == 'B']['age'].dropna()
hispanic_ages = df[df['race'] == 'H']['age'].dropna()
# Filter age data for white victims
white_ages = df[df['race'] == 'W']['age'].dropna()
# Plotting the histogram for white victims
plt.figure(figsize=(8, 6))
plt.hist(white_ages, bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution of White Victims')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
hispanic_ages = df[df['race'] == 'H']['age'].dropna()
# Plotting histogram for Hispanic individuals
plt.figure(figsize=(8, 6))
# Plotting the age distribution for Hispanic victims
plt.hist(hispanic_ages, bins=30, color='green', alpha=0.7)
plt.title('Age Distribution of Hispanic Victims')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
# Filter age data for black individuals
black_ages = df[df['race'] == 'B']['age'].dropna()
# Plotting the histogram for the age distribution of black victims
plt.figure(figsize=(8, 6))
plt.hist(black_ages, bins=30, color='red', alpha=0.7)
plt.title('Age Distribution of Black Victims')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(white_ages, bins=30, alpha=0.5, color='blue', label='White')
plt.hist(black_ages, bins=30, alpha=0.5, color='red', label='Black')
plt.hist(hispanic_ages, bins=30, alpha=0.5, color='green', 
label='Hispanic')
plt.title('Age Distribution of Victims by Race')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
import numpy as np
num_trials = 1000
# Store simulated ages from the dataset
simulated_ages = df['age'].dropna().values
# Perform Monte Carlo simulation for average age
simulated_average_ages = np.mean(np.random.choice(simulated_ages, 
(len(simulated_ages), num_trials)), axis=0)
# Print the estimate
print(f"Monte Carlo estimate of average age of victims: 
{average_age_estimate:.2f} years")
white_ages = df[df['race'] == 'W']['age'].dropna().values
black_ages = df[df['race'] == 'B']['age'].dropna().values
num_trials = 1000
simulated_differences = []
for _ in range(num_trials):
 white_sample = np.random.choice(white_ages, len(white_ages))
 black_sample = np.random.choice(black_ages, len(black_ages))
 difference = np.mean(white_sample) - np.mean(black_sample)
 simulated_differences.append(difference)
mean_difference = np.mean(simulated_differences)
print(f"Monte Carlo estimate of difference in mean ages (White - Black): 
{mean_difference:.2f} years")
import matplotlib.pyplot as plt
# Assuming 'mean_difference' holds the Monte Carlo estimate of the 
difference in mean ages
plt.figure(figsize=(8, 6))
plt.hist(simulated_differences, bins=30, color='skyblue', 
edgecolor='black')
plt.axvline(mean_difference, color='red', linestyle='dashed', linewidth=2, 
label=f'Mean Diff: {mean_difference:.2f}')
plt.title('Monte Carlo Simulation: Difference in Mean Ages (White -
Black)')
plt.xlabel('Difference in Mean Ages')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# Q. What is the distribution of police shootings by gender and race?
# Create a pivot table to count shootings by gender and race
shootings_by_gender_race = data.groupby(['gender', 
'race'])['id'].count().unstack()
# Create a stacked bar chart
shootings_by_gender_race.plot(kind='bar', stacked=True)
plt.title('Distribution of Police Shootings by Gender and Race')
plt.xlabel('Gender')
plt.ylabel('Number of Shootings')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])
# Extract the year from the 'date' column
data['year'] = data['date'].dt.year
# Group by year and count the number of shootings
shootings_by_year = data.groupby('year')['id'].count()
# Plotting the number of shootings year-wise
plt.figure(figsize=(10, 6))
shootings_by_year.plot(kind='bar', color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Shootings')
plt.title('Number of Police Shootings Year-wise')
plt.show()
print("Q. Finding the year with the maximum shootings.")
max_year = shootings_by_year.idxmax()
print(f" The year with the maximum shootings is {max_year} with 
{max_shootings} shootings.")
import pandas as pd
import matplotlib.pyplot as plt
# Answer questions related to body_camera
total_incidents = len(df)
body_camera_incidents = df['body_camera'].sum()
percentage_with_body_camera = (body_camera_incidents / total_incidents) * 
100
print(f"Total incidents: {total_incidents}")
print(f"Incidents with body camera: {body_camera_incidents}")
print(f"Percentage of shootings with body camera: 
{percentage_with_body_camera:.2f}%")
# Count the occurrences where body cameras were off (marked as FALSE)
body_cam_off_count = df[df['body_camera'] == False]['body_camera'].count()
# Total number of instances
total_instances = len(df)
# Calculate the percentage of body cameras that were off
percentage_body_cam_off = (body_cam_off_count / total_instances) * 100
print(f"Percentage of shootings without body cameras: 
{percentage_body_cam_off:.2f}%")
# Plot a bar chart to visualize the data
labels = ['Without Body Camera', 'With Body Camera']
values = [total_incidents - body_camera_incidents, body_camera_incidents]
plt.bar(labels, values, color=['red', 'green'])
plt.title('Police Shootings with and without Body Camera')
plt.xlabel('Body Camera Presence')
plt.ylabel('Number of Incidents')
plt.show()
# Plot histogram for armed_with
plt.figure(figsize=(12, 6))
sns.countplot(x='armed_with', data=data)
plt.title('Distribution of Armed With')
plt.xticks(rotation=45)
plt.show()
print("Q. Is having a weapon increases the chance of a shooting?")
armed_shootings = data[data['armed_with'].notna()]
unarmed_shootings = data[data['armed_with'].isna()]
weapon_chance = len(armed_shootings) / len(unarmed_shootings)
print(f" The chance of a shooting when armed with a weapon is 
{weapon_chance:.2f} times higher.")
# Q. What are the most common threat types in police shootings?
# Create a histogram of threat types
plt.figure(figsize=(12, 6))
sns.countplot(x='threat_type', data=data)
plt.title('Distribution of Threat Type')
plt.show()
import pandas as pd
# Your code for sorting the data (state_crime_ranking) goes here
top_n_states = 10 # Change this to display a different number of states
top_states = state_crime_ranking.head(top_n_states)
bottom_states = state_crime_ranking.tail(top_n_states)
# Create DataFrames for top and bottom states
top_states_df = pd.DataFrame(top_states)
bottom_states_df = pd.DataFrame(bottom_states)
# Function to format a DataFrame with borders
def format_dataframe_with_borders(df):
 table = df.to_string(index=False)
 table_with_borders = f"+{'-' * (len(table.splitlines()[0]) - 2)}+\n" 
# Create top border
 table_with_borders += f"{table}\n" # Add the table content
 table_with_borders += f"+{'-' * (len(table.splitlines()[0]) - 2)}+" # 
Create bottom border
 return table_with_borders
# Display top and bottom states in tables with borders
print("Top States by Crime Rate:")
print(format_dataframe_with_borders(top_states_df))
print("\nBottom States by Crime Rate:")
print(format_dataframe_with_borders(bottom_states_df))
import seaborn as sns
# Group by state and count shootings
state_shootings = 
data.groupby('state').size().reset_index(name='shootings_count')
# Plot a bar chart
plt.figure(figsize=(17, 8))
sns.barplot(x='state', y='shootings_count', data=state_shootings)
plt.title('Police Shootings Distribution by State')
plt.xlabel('State')
plt.ylabel('Number of Shootings')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import DBSCAN
# Assuming 'data' is your DataFrame with 'latitude' and 'longitude' 
columns
# Filter data to include only US coordinates
us_data = data[(data['latitude'] >= 24.396308) & (data['latitude'] <= 
49.384358) & 
 (data['longitude'] >= -125.000000) & (data['longitude'] <= 
-66.934570)]
# Select latitude and longitude columns for clustering
X = us_data[['latitude', 'longitude']]
# Perform DBSCAN clustering
db = DBSCAN(eps=0.2, min_samples=5).fit(X)
us_data['cluster'] = db.labels_
# Create a cartopy plot to visualize clusters on the map
plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
# Scatter plot with cluster colors
colors = us_data['cluster']
ax.scatter(us_data['longitude'].values, us_data['latitude'].values, 
c=colors, cmap='viridis', marker='o', s=20)
plt.title('Clustering of Police Shooting Incidents in the US')
plt.show()
#import geopandas as gpd
#from shapely.geometry import Point
# Example DataFrame
#data = {'City': ['CityA', 'CityB', 'CityC'],'Latitude': [40.7128, 
34.0522, 41.8781],'Longitude': [-74.0060, -118.2437, -87.6298]}
# Create a GeoDataFrame from the DataFrame with Point geometries
#geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
#gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
#gdf = gpd.read_file('C:\Users\meggn\Downloads\USA_adm.shp')
import folium
from folium.plugins import HeatMap
# Assuming you have a pandas DataFrame named 'data' with 'latitude' and 
'longitude' columns
data = data.dropna(subset=['latitude', 'longitude'])
# Create a base map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4) # Centered in 
the US
# Create a HeatMap layer
heat_data = [[row['latitude'], row['longitude']] for _, row in 
data.iterrows()]
HeatMap(heat_data).add_to(m)
# Save the map as an HTML file
m.save('fatal_police_shootings_heatmap.html')
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, 
classification_report
data = pd.read_excel(r'C:\Users\meggn\Documents\MTH 522\Project 2\fatalpolice-shootings-data-web.xlsx') 
# Create a binary target column based on the 'manner_of_death' column
data['target'] = (data['manner_of_death'] == 'shot').astype(int)
# Select features for X
X = data[['age', 'signs_of_mental_illness']]
# Impute missing values in the 'age' column with the mean
imputer = SimpleImputer(strategy='mean')
X['age'] = imputer.fit_transform(X[['age']])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['target'], 
test_size=0.2, random_state=42)
# Create a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate and print the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion)
# Generate a classification report
report = classification_report(y_test, y_pred)
print("Logistic Regression Model Accuracy:", accuracy)
print("Classification Report:")
print(report)
