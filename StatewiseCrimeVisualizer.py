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
