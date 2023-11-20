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
