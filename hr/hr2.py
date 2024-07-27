import pandas as pd
import os

def get_heartrate_for_date(hr, start, end):
    hr = hr[hr["startDate"] >= start]
    hr = hr[hr["endDate"] <= end]
    return hr

workout_idx_from_end = 1

# /mnt/d/studia/export/export.xml
hr_data_path = os.path.join('/mnt','d','studia','export', 'HeartRate.csv')
workout_data_path = os.path.join('/mnt','d','studia','export', 'Workout.csv')

# 2020-01-01 00:22:47 +0300
heartrate_data = pd.read_csv(hr_data_path, date_format='%Y-%m-%d %H:%M:%S %z',
                             parse_dates=['creationDate','startDate','endDate'],
                             dtype={
                                'sourceName': 'string',
                                'sourceVersion': 'string',
                                'device': 'string',
                                'type': 'string',
                                'unit': 'string',
                                'creationDate': 'string',
                                'startDate': 'string',
                                'endDate': 'string',
                                'value': 'float64',
                            })

workout_data = pd.read_csv(workout_data_path, date_format='%Y-%m-%d %H:%M:%S %z', 
                           parse_dates=['creationDate','startDate','endDate'],
                           dtype={
                                'sourceName': 'string',
                                'sourceVersion': 'string',
                                'device': 'string',
                                'creationDate': 'string',
                                'startDate': 'string',
                                'endDate': 'string',
                                'workoutActivityType': 'string',
                                'duration': 'float64',
                                'durationUnit': 'string',
                                'totalDistance': 'float64',
                                'totalDistanceUnit': 'string',
                                'totalEnergyBurned': 'float64',
                                'totalEnergyBurnedUnit': 'string'
                            })


# Extract heartrate statistics for certain workout
last_workout = workout_data.iloc[[-workout_idx_from_end]]
heartrate_workout = get_heartrate_for_date(heartrate_data, last_workout["startDate"].item(), 
                                           last_workout["endDate"].item())

# value is numeric, NaN if fails
heartrate_workout['value'] = pd.to_numeric(heartrate_workout['value'], errors='coerce')

# some records do not measure anything, just count occurences
# filling with 1.0 (= one time) makes it easier to aggregate
heartrate_workout['value'] = heartrate_workout['value'].fillna(1.0)

print(heartrate_workout.head())

heartrate_workout.to_csv('./hr_results/gt/hr.csv', index=False)
