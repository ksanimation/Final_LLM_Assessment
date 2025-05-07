import pandas as pd
import numpy as np
import math as math
import seaborn as sns
from SentimentLabelsDataFrame import dfSentiment as df
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# __Task 2__ Exploratory Data Analysis (EDA)

##Check imported df print(df.head(5))

# df.describe()
# Output: RangeIndex: 2191 entries, 0 to 2190. Used to describe dataframe

# Change Sentiment to category for plots
df['Sentiment'] = df['Sentiment'].astype('category')

# Find data types of each column
# print(df.info())
# Output: Shows Data Type of columns. Subject, body, and date are object types, while date is datetime64[ns]. Sentiment is a category type.


# Check for duplicates
# print(df.loc[df.duplicated()]
# No duplicates found

# Check for null values
# print(df.isna().sum())
# No null values found

#Plots and findings
"""
# Histogram for Sentiment and overall count
sns.histplot(df['Sentiment'])
plt.show()

# Boxplot for distribution of Sentiment acrosss dates
sns.boxplot(x='date', y='Sentiment', data=df)
plt.show() 

# Countplot for the porportion of sentiment in emails sent by each employee
sns.countplot(x='from', data=df, hue='Sentiment')
plt.show()

# Extract month and year from date for Month-Year feature
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M').astype(str)
sorted_dates = sorted(df['month_year'].unique(), key=lambda x: pd.Period(x, freq='M'))
df['month_year'] = pd.Categorical(df['month_year'], categories=[str(d) for d in sorted_dates], ordered=True)

# Countplot for the distribution of sentiment over time by month
sns.countplot(x='month_year', data=df, hue='Sentiment')
plt.show()
"""

# __Task 3__ Employee Score Calculation

# Separate into months
df['month_year'] = df['date'].dt.to_period('M')
uniquemonths = df['month_year'].unique()

# Dictionary to store scores by month
monthly_scores = {}

# Iterate through each month and calculate scores. Store scores in employee scores within monthly_scores

for month in uniquemonths:
    monthly_df = df[df['month_year'] == month]
    employee_score = dict.fromkeys(monthly_df['from'].unique(), 0)
    
    for i in range(len(monthly_df)):

        # Use index with iloc to get the value of the column
        sentiment = monthly_df.iloc[i]['Sentiment']
        sender = monthly_df.iloc[i]['from']
        
        # Neutral is ignored and treated as 0 while other sentiments get a numerical value added or subtracted as a value to the dictionary monthly_scores
        if sentiment == 'Positive':
            employee_score[sender] = employee_score.get(sender, 0) + 1
        elif sentiment == 'Negative':
            employee_score[sender] = employee_score.get(sender, 0) - 1
        
    # Store in monthly_scores dictionary created earlier
    monthly_scores[str(month)] = employee_score

# __Task 4__ Employee Ranking

# Create lists to store top positive and negative employees
top_positive_employees = []
top_negative_employees = []

for month in monthly_scores:
    # Sort the employee scores in descending order to get top scores
    sorted_scores = sorted(monthly_scores[month].items(), key=lambda x: x[1], reverse=True)
    
    # Get top 5 positive and negative employees in alphabetical order
    top_positives = sorted(sorted_scores[:5])
    top_negatives = sorted(sorted_scores[-5:])

 
     # Append to lists for each month in format to put into table
    for name, score in top_positives:
        top_positive_employees.append({'Month': month, 'Employee': name, 'Score': score})
    for name, score in top_negatives:
        top_negative_employees.append({'Month': month, 'Employee': name, 'Score': score})


# Create DataFrames to make tables for data and sort by month
top_positive_df = pd.DataFrame(top_positive_employees)
top_positive_df['Month'] = pd.to_datetime(top_positive_df['Month'])
top_positive_df = top_positive_df.sort_values(by='Month')

top_negative_df = pd.DataFrame(top_negative_employees)
top_positive_df['Month'] = pd.to_datetime(top_negative_df['Month'])
top_positive_df = top_negative_df.sort_values(by='Month')

# Export Visualization
facetgrid_pos = sns.FacetGrid(top_positive_df, col="Month", col_wrap=3, height=4, sharex=False, sharey=False)
facetgrid_pos.map_dataframe(sns.barplot,x="Employee",y="Score", dodge=False)
facetgrid_pos.set_titles(col_template="{col_name}")
facetgrid_pos.set_axis_labels("Employee", "Score")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle("Top 5 Positive Employees Per Month", fontsize=16)
plt.show() 

facetgrid_neg = sns.FacetGrid(top_negative_df, col="Month", col_wrap=3, height=4, sharex=False, sharey=False)
facetgrid_neg.map_dataframe(sns.barplot,x="Employee",y="Score", dodge=False)
facetgrid_neg.set_titles(col_template="{col_name}")
facetgrid_neg.set_axis_labels("Employee", "Score")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle("Top 5 Negative Employees Per Month", fontsize=16)
plt.show() 


# __Task 5__ Flight Risk Identification

# Ensure date column is in datetime and set as index for rolling function
df['date'] = pd.to_datetime(df['date'])  
df.set_index('date', inplace=True)

# Ensures dates are in order for .rolling function
df.sort_index(inplace=True)

# Store rolling 30 day count of each sentiment in rolling_days variable
rolling_days = df['Sentiment'].rolling('30D').count()

# Create list to store flight risk employees
flightrisk = []

for i in range(len(df)):
    
    # Use iloc to access row in dataframe
    sentiment = df.iloc[i]['Sentiment']
    sender = df.iloc[i]['from']
    count = rolling_days.iloc[i] 

    # If the sentiment is negative and if the rolling days count is greater than 4, then it adds employee to flight risk list
    if sentiment == 'Negative' and count >= 4 and sender not in flightrisk:
        flightrisk.append({'Employee': sender, 'RollingCount': count})

# Convert list to DataFrame
flightrisk_df = pd.DataFrame(flightrisk)

# Export Visualization
flightrisk_df.sort_values(by='RollingCount', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='RollingCount', y='Employee', data=flightrisk_df)
plt.title('Flight Risk Employees and Message Count')
plt.xlabel('30-Day Message Count')
plt.ylabel('Employee')
plt.tight_layout()
plt.show()

# __Task 6__ Predictive Modeling

# Ensure 'date' is in datetime format
df['date'] = pd.to_datetime(df['date'])  

# Assign Sentiment to numerical values for analysis
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map)

#Extract month and year from date for Month-Year feature
df['Month_Year'] = df['date'].dt.to_period('M')

# Sort dates
df = df.sort_values(['date'])


# Create some features for model. Can add more features later
df['Message_Frequency'] = df.groupby(df['from'])['date'].transform('count') 
df['Day_of_Week'] = df['date'].dt.dayofweek
df['Prev_Sentiment_SameSender'] = df.groupby('from')['Sentiment_Score'].shift(1).fillna(0)
df['Prev_Sentiment_2_SameSender'] = df.groupby('from')['Sentiment_Score'].shift(2).fillna(0)
df['Prev_Sentiment_3_SameSender'] = df.groupby('from')['Sentiment_Score'].shift(3).fillna(0)
df['Message_Frequency_MonthYear'] = df.groupby(df['from'])['Month_Year'].transform('count') 
df['Time_Since_Last_Message'] = df.groupby('from')['date'].diff().dt.days.fillna(0) 
df['Msg_Count_Rolling30'] = (df.groupby('from')['Sentiment_Score'].transform(lambda x: x.rolling(window=30, min_periods=1).count()))

df['Message_Length'] = df['body'].str.len()
df['Word_Count'] = df['body'].str.split().str.len()
df['Cumulative_Sentiment_Avg'] = (df.groupby('from')['Sentiment_Score'].expanding().mean().reset_index(level=0, drop=True))



# X  are features and Y is the target variable
X = df[['Cumulative_Sentiment_Avg']]
y = df['Sentiment_Score']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Validate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Print coefficients of the model for analysis
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

""" Measure of good MSE is MSE<.22 with positive R-squared.
Message_frequency had little predictive value with high MSE and negative R-squared. Positive Correlation with Sentiment.
Day_of week had little predictive value with high MSE and negative R-squared. Positive Correlation with Sentiment.
Prev_Sentiment_SameSender had little predictive value with high MSE and negative R-squared Negative correlation with Sentiment. Same with 2 and 3 separately and combined together.
Msg_Count_Rolling30 had little predictive value with high MSE and negative R-squared. Positive correlation with Sentiment.

Created Message_Frequency_MonthYear to see if it had a better predictive value than Message_Frequency for specific days. I ideally wanted the Month_Year to
return more frequencies per employee. However results were the same as Message_Frequency. 

Time_Since_Last_Message had little predictive value with high MSE and negative R-squared. Negative correlation with Sentiment.
Message_Length had a much better predictive value with a MSE of .31 and positive R-squared pf pf  0.065. Positive correlation 0.0005 with Sentiment.
Word_Count had a much better predictive value with a MSE of .31 and positive R-squared pf pf  0.075. Positive correlation 0.003 with Sentiment
Cumulative_Sentiment_Avg had a much better predictive value with a MSE of .31 and positive R-squared pf pf  0.024. Positive correlation 1.138899 with Sentiment."""
