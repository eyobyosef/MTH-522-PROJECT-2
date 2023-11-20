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
