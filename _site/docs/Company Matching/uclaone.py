"""
Purpose: UCLA ONE Project

Author: Odeya Russo
    
Date: 02/20/2024

Notes:
    Files:
        - Companies file: all the companies and their IDS listed in CRM
        - User file: all the constituent information listed in the UCLA ONE platform
    Goal:
        - Match the company column in both datasets using FuzzyWuzzy library
        - Can look at the most popular company names first
        
    # STEP 1:
        # Convert data to tall format
        # 02/26/24: DONE

    # MATCHING PROBLEM:
        # create one list for unique company names from ucla one
        # create one list for unique company names from CRM
        # create an algorithm using either FuzzyWuzzy or NLTK library to calculate the similarity scores between company lists

    # UPDATING PROBLEM:
        # Which platform should be updated, CRM or UCLA ONE?
        # Maybe look into the platform with the most recently inputed company title
        # Use fields like inference date, last activity date, etc to decide whether it is worth to update

    # WHAT TO ANALYZE PROBLEM:
        # Narrow down to users who have had activity on UCLAONE in the last 6 months or year
        # Can even narrow it down to those that logged in more than once in the last year
        # Can filter users based on fields like last activity date, employment update date, etc



"""

import pyIS.toolbox as tbx
import pandas as pd
bbdw = tbx.bbdw_connect(username='orusso')

from pyIS.Engagement_Summary.engagement_summary import EngagementSummary
egs = EngagementSummary(bbdw_username='orusso')

from pyIS.Quick_Reports import quick_reports
qr = quick_reports.QuickReports('orusso')


#%%
###############################################################################
# Read in data files and explore datasets
###############################################################################

# companies data file
companies_df = pd.read_csv("companies.csv")
companies_columns = companies_df.columns.tolist()
#tbx.quick_counts(companies_df, 'Company name')
#na_count_companies = sum([True for idx,row in companies_df.iterrows() if any(row.isnull())])


# user data file
user_df = pd.read_csv("active-user-list.csv")
user_columns = user_df.columns.tolist()
#tbx.quick_counts(user_df, 'Company')
#na_count_user = sum([True for idx,row in user_df.iterrows() if any(row.isnull())])


#%%
###############################################################################
# Changing the first work experience column names to have a zero at the end
# (This helps transform the data from wide to long format)
###############################################################################

first_experience_columns = ['Company (Work Experience)', 'Position (Work Experience)',
                            'Industry (Work Experience)', 'Start date (Work Experience)',
                            'End date (Work Experience)']

new_column_names = ['Company (Work Experience)(0)', 'Position (Work Experience)(0)',
                    'Industry (Work Experience)(0)', 'Start date (Work Experience)(0)',
                    'End date (Work Experience)(0)']

# Rename the columns
user_df.rename(columns=dict(zip(first_experience_columns, new_column_names)), inplace=True)

# Reset user_columns variable to new user_df columns
user_columns = user_df.columns.tolist()


#%%
###############################################################################
# Storing only the important columns in the user_df dataframe
###############################################################################

# List of columns containing basic user information
user_basic_info_cols = ["Unique Id", "Registration date", "Update date", "Last login date"]

# List of columns containing all work experiences
user_work_info_cols = user_columns[350:len(user_columns) + 1]

# Columns to include in the test dataframe
selected_columns = user_basic_info_cols + user_work_info_cols

# Rows that have at least one value for the work experience columns
work_experience_rows = user_df[user_df[user_work_info_cols].notna().any(axis=1)]

# The index of the rows that have at least one work experience
selected_rows_index = work_experience_rows.index


        
#%%
###############################################################################
# Attempting to transform test data from wide to long format
# 02/23/24: It works on the test dataframe with only the first 5 observations
# 02/26/24: It worked on the entire dataset, yay!!!
###############################################################################

# Test Dataframe consisting of all users that have at least one work experience
test_df = user_df.loc[selected_rows_index, selected_columns]

# Use the melt function to reshape the DataFrame
test_melted_df = pd.melt(test_df, id_vars=['Unique Id', 'Registration date', 'Update date', 'Last login date'],
                         var_name='variable', value_name='value')

# Extract the number from the 'variable' column to get the work experience number
test_melted_df['experience_number'] = test_melted_df['variable'].str.extract(r'\((\d+)\)')

# Pivot the DataFrame to get the desired format
test_long_df = test_melted_df.pivot_table(index=['Unique Id', 'Registration date', 'Update date', 'Last login date', 'experience_number'],
                                          columns='variable', values='value', aggfunc='first').reset_index()



# The following for loop is used to extract only the non-na values from the work experience columns for each row
selected_values_list = []

for index, row in test_long_df.iterrows():
    
    # store the work experience number
    experience_number = row['experience_number']
    # create a string of the work experience number to match the syntax of the work experience columns
    experience_string = "(" + str(experience_number) + ")"
    
    # Use list comprehension to only select columns that end with the experience number string
    selected_cols = [column_name for column_name in test_long_df.columns.tolist() if column_name.endswith(experience_string)]
    
    # Define a list containing some basic user column names
    user_basic_info_cols = ['Unique Id', 'Registration date', 'Update date', 'Last login date', 'experience_number']
    
    # Combine basic info columns with selected columns
    final_cols = user_basic_info_cols + selected_cols
    
    # Select the desired values by filtering the dataframe with desired rows and columns
    selected_values_row = test_long_df.loc[index, final_cols].values
    
    # Add the selected values to their own row in the selected_values list
    selected_values_list.append(selected_values_row)
 
    
# Creat a list of the column names of the tall dataframe in the correct order
end_columns = user_basic_info_cols + ['Company (Work Experience)', 'End date (Work Experience)',
                                      'Industry (Work Experience)', 'Position (Work Experience)', 
                                      'Start date (Work Experience)']

# Create a dataframe with the filtered values and the column names
selected_values_df = pd.DataFrame(selected_values_list, columns=end_columns)

# Order the columns of the dataframe to reflect more logical job experience order
order_selected_values_df = selected_values_df[['Unique Id', 
                                               'Registration date', 
                                               'Update date', 
                                               'Last login date', 
                                               'experience_number', 
                                               'Company (Work Experience)',
                                               'Position (Work Experience)',
                                               'Industry (Work Experience)',
                                               'Start date (Work Experience)',
                                               'End date (Work Experience)']]

#%%
###############################################################################
# Create a criteria to decide which observations in the data to keep based on 
# user activity
###############################################################################

# Creating dataframe with only neccessary columns
user_date_df = order_selected_values_df[['Update date', 'Last login date']].copy()

# Converting 'Update date' and 'Last login date' columns to date format
user_date_df.loc[:, 'Update date'] = pd.to_datetime(user_date_df['Update date']).dt.date
user_date_df.loc[:, 'Last login date'] = pd.to_datetime(user_date_df['Last login date']).dt.date


# Creating a table displaying the min and max dates for each date column
activity_columns = ['Update date', 'Last login date']
min_dates = user_date_df[activity_columns].min()
max_dates = user_date_df[activity_columns].max()

activity_range_summary = pd.DataFrame({
    'Minimum Date': min_dates,
    'Maximum Date': max_dates
})

print(activity_range_summary)

#%%
###############################################################################
# SELECTION CRITERTION:
## Last activity date in user dataframe: 2023-03-22
## Only select users who updated their UCLAONE profile in the last year of all dates
## Only select users who last logged in to their UCLAONE profile in the last year of all dates
###############################################################################

# Creating an active_user_df based on selection cirteria
active_user_df = order_selected_values_df[
    (order_selected_values_df['Update date'] >= '2022-03-22') &
    (order_selected_values_df['Update date'] <= '2023-03-22') &
    (order_selected_values_df['Last login date'] >= '2022-03-22') &
    (order_selected_values_df['Last login date'] <= '2023-03-22')
]

# Print out active_user_df information
print(active_user_df.info())


#%%
###############################################################################
# Comparing most popular company names from user_df and companies_df
###############################################################################

# Extracting the unique companies and their value counts for both UCLAONE and CRM company names
user_companies_counts = active_user_df["Company (Work Experience)"].value_counts()
crm_companies_counts = companies_df["Company name"].value_counts()

print(f"UCLAONE User Company Counts:\n{user_companies_counts}\n")
print(f"CRM Company Counts:\n{crm_companies_counts}\n")






