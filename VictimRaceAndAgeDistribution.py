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
