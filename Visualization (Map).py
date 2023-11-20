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
