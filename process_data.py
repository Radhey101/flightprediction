import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
import torch
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F

def preprocess_of_data(df):

    # clean of data
    df.isnull().sum()
    df.dropna(inplace=True)
    df.isnull().sum()

    # print dublicate value
    df.duplicated()
    df.drop_duplicates(inplace=True)

    df['total_stop'] = df['Stop'].apply(get_total_stop)

    ## Convert date data into formated and clean it

    df['Departure_Date'] = pd.to_datetime(df['DepartureDate'], format='%d%m%Y')
    df['SearchDate'] = df['SearchDate'].apply(lambda x: x.split(" ")[0])

    df['SearchDate'] = pd.to_datetime(df['SearchDate'])
    df['search_date_day'] = df['SearchDate'].dt.day
    df['search_date_month'] = df['SearchDate'].dt.month
    df['search_date_year'] = df['SearchDate'].dt.year

    df['days_until_departure'] = (df['Departure_Date'] - df['SearchDate']).dt.days

    # Convert duration to total minutes
    df['duration_mins'] = df['Duration'].apply(convert_duration_to_minutes)

    df['festival_season'] = df['Departure_Date'].apply(check_festival_season)

    df['dept_day_of_week'] = df['Departure_Date'].dt.dayofweek
    df['dept_day_of_year'] = df['Departure_Date'].dt.dayofyear
    df['dept_is_weekend'] = df['Departure_Date'].isin([5, 6]).astype(int)

    # Apply function to DataFrame column
    df["departure_segment"] = df["Departure"].apply(get_time_segment)
    df["arrival_segment"] = df["Arrival"].apply(get_time_segment)
    df["RouteKey"] = df["Source"] + "-" + df["Destination"]

    df["Price"] = df["Price"].str.replace("₹", "", regex=True)
    df["Price"] = df["Price"].str.replace(",", "", regex=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # find outlier
    # make actual values
    df["Actual_Price"] = df["Price"]
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    iqr = int(Q3) - int(Q1)
    maxnum = int(Q3) + 1.5 * int(iqr)
    minnum = int(Q1)- 1.5 * int(iqr)

    np.where(df["Price"] > maxnum, df["Price"].median(), df["Price"])

    # delete colomns already modify and encoded
    df.drop('Departure', axis=1, inplace=True)
    df.drop('Arrival', axis=1, inplace=True)
    df.drop('Duration', axis=1, inplace=True)
    #df.drop('DepartureDate', axis=1, inplace=True)
    #df.drop('SearchDate', axis=1, inplace=True)
    df.drop('Day', axis=1, inplace=True)
    df.drop('Stop', axis=1, inplace=True)
    df.drop('search_date_year', axis=1, inplace=True)

    df["SearchDate"] = pd.to_datetime(df["SearchDate"])
    df["DepartureDate"] = pd.to_datetime(df["DepartureDate"], format="%d%m%Y", errors="coerce")

    return df

# Function to classify time segments
def get_time_segment(time_data):
    hour = int(time_data.split(':')[0].strip())
    if 0 <= hour < 5:
        return "Late_Night"
    elif 5 <= hour < 8:
        return "Early_Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# Apply function to DataFrame

# Define festival date ranges (customize based on year and known dates)
def check_festival_season(date_to_check):
    festival_seasons = [
        ('Diwali', datetime(2025, 10, 20), datetime(2025, 10, 26)),
        ('Dussehra', datetime(2025, 10, 1), datetime(2025, 10, 5)),
        ('Christmas', datetime(2025, 12, 24), datetime(2025, 12, 26)),
        ('Holi', datetime(2025, 3, 13), datetime(2025, 3, 14)),
        ('Independence Day', datetime(2025, 8, 15), datetime(2025, 8, 15)),
        ('New Year', datetime(2025, 12, 31), datetime(2026, 1, 1)),
        ('Holi_26', datetime(2026, 3, 1), datetime(2026, 3, 6)),
    ]
    for festival_name, start_date, end_date in festival_seasons:
        if start_date <= date_to_check <= end_date:
            return 1
    return 0


def convert_duration_to_minutes(duration):
    try:
        if not isinstance(duration, str) or duration.strip() == "":
            return 0

        duration = duration.strip().lower()
        days = hours = minutes = 0

        # Extract days
        if 'd' in duration:
            parts = duration.split('d')
            days = int(parts[0].strip())
            duration = parts[1].strip() if len(parts) > 1 else ""

        # Extract hours
        if 'h' in duration:
            parts = duration.split('h')
            hours = int(parts[0].strip())
            duration = parts[1].strip() if len(parts) > 1 else ""

        # Extract minutes
        if 'm' in duration:
            minutes = int(duration.replace('m', '').strip())

        return days * 1440 + hours * 60 + minutes
    except Exception as e:
        print(f"Warning: Failed to parse duration '{duration}' – {e}")
        return 0



def get_total_stop(x):
    if(x.isnumeric()):
            return int(x)
    else:
        update_str = x[0]
        if(x == 'Non stop' or update_str == 'u') :
            return 0
        else:
            return int(update_str)


