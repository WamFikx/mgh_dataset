import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# Load data
try:
    blue_team_df = pd.read_csv('blue_team_sales.csv')
    red_team_df = pd.read_csv('red_team_sales.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    exit()

# Standardize column names (some CSVs use 'money' instead of 'amount')
if 'money' in blue_team_df.columns:
    blue_team_df = blue_team_df.rename(columns={'money': 'amount'})
if 'money' in red_team_df.columns:
    red_team_df = red_team_df.rename(columns={'money': 'amount'})

# Function to clean red team time values
def clean_red_time(time_str):
    try:
        # Handle formats like "15:50.5" or invalid "46:33.0"
        parts = str(time_str).split(':')
        hours = int(parts[0]) % 24  # Ensure valid hour
        minutes = int(float(parts[1]))  # Handle decimal minutes
        return f"{hours:02d}:{minutes:02d}"
    except:
        return None

# Parse datetimes with proper formats
try:
    # Blue team: "13/02/2025 07:55" (day/month/year)
    blue_team_df['datetime'] = pd.to_datetime(
        blue_team_df['datetime'],
        dayfirst=True,
        format='%d/%m/%Y %H:%M'
    )
    
    # Red team: needs date + time combination
    red_team_df['clean_time'] = red_team_df['datetime'].apply(clean_red_time)
    red_team_df['datetime'] = pd.to_datetime(
        red_team_df['date'] + ' ' + red_team_df['clean_time'],
        errors='coerce'
    )
    
    # Remove rows with invalid datetimes
    blue_team_df = blue_team_df.dropna(subset=['datetime'])
    red_team_df = red_team_df.dropna(subset=['datetime'])
    
except Exception as e:
    print(f"Error parsing datetime: {e}")
    exit()

# Display cleaned data
print("Blue Team Data (first 5 rows):")
print(blue_team_df.head())
print("\nRed Team Data (first 5 rows):")
print(red_team_df.head())

# Basic Revenue Analysis
blue_revenue = blue_team_df['amount'].sum()
red_revenue = red_team_df['amount'].sum()

print(f"\nTotal Revenue:")
print(f"Blue Team: ${blue_revenue:.2f}")
print(f"Red Team: ${red_revenue:.2f}")

if blue_revenue > red_revenue:
    diff = blue_revenue - red_revenue
    print(f"Blue Team leads by ${diff:.2f}")
elif red_revenue > blue_revenue:
    diff = red_revenue - blue_revenue
    print(f"Red Team leads by ${diff:.2f}")
else:
    print("Both teams have equal revenue")

# Top Products Analysis
print("\nTop 3 Products by Team:")
for team, df in [('Blue', blue_team_df), ('Red', red_team_df)]:
    top_products = df['coffee_name'].value_counts().nlargest(3)
    print(f"\n{team} Team Top Products:")
    print(top_products)

# Time-based Analysis
def add_time_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].ge(5).astype(int)
    df['date'] = df['datetime'].dt.date
    return df

blue_team_df = add_time_features(blue_team_df)
red_team_df = add_time_features(red_team_df)

# Hourly Sales Patterns
print("\nPeak Sales Hours:")
for team, df in [('Blue', blue_team_df), ('Red', red_team_df)]:
    hourly_sales = df.groupby('hour')['amount'].sum()
    peak_hour = hourly_sales.idxmax()
    peak_sales = hourly_sales.max()
    print(f"{team} Team peak at {peak_hour}:00 - ${peak_sales:.2f}")

# Weekend vs Weekday Analysis
print("\nWeekend vs Weekday Performance:")
for team, df in [('Blue', blue_team_df), ('Red', red_team_df)]:
    sales_by_day_type = df.groupby('is_weekend')['amount'].agg(['sum', 'mean'])
    sales_by_day_type.index = ['Weekday', 'Weekend']
    print(f"\n{team} Team:")
    print(sales_by_day_type)

# Visualization
plt.figure(figsize=(12, 6))
for team, df, color in [('Blue', blue_team_df, 'blue'), ('Red', red_team_df, 'red')]:
    daily_sales = df.groupby('date')['amount'].sum()
    plt.plot(daily_sales.index, daily_sales.values, label=f'{team} Team', color=color)

plt.title('Daily Sales Comparison')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.show()

# Statistical Comparison
t_stat, p_val = stats.ttest_ind(blue_team_df['amount'], red_team_df['amount'])
print(f"\nStatistical Comparison (t-test):")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")
if p_val < 0.05:
    print("Difference is statistically significant")
else:
    print("No significant difference found")

# Machine Learning Models (Optional)
# Daily sales prediction model example
def train_sales_model(df, team_name):
    daily = df.groupby('date')['amount'].sum().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['month'] = daily['date'].dt.month
    
    X = daily[['day_of_week', 'month']]
    y = daily['amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n{team_name} Team Sales Prediction Model:")
    print(f"R-squared: {model.score(X_test, y_test):.3f}")
    return model
