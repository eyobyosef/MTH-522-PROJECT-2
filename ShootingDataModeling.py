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
