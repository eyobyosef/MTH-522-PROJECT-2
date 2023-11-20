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
