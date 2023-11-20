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
plt.show(
