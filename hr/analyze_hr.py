import pandas as pd
import os
import xml.etree.ElementTree as ET

# create element tree object
data_path = os.path.join('/mnt','d','studia','export', 'export.xml')
tree = ET.parse(data_path) 
# for every health record, extract the attributes
root = tree.getroot()
record_list = [x.attrib for x in root.iter('Record')]

record_data = pd.DataFrame(record_list)

print(record_data.head())

# proper type to dates
for col in ['creationDate', 'startDate', 'endDate']:
    record_data[col] = pd.to_datetime(record_data[col], utc=True)

# shorter observation names
record_data['type'] = record_data['type'].str.replace('HKQuantityTypeIdentifier', '')
record_data['type'] = record_data['type'].str.replace('HKCategoryTypeIdentifier', '')
print(record_data.tail())

def get_heartrate_for_date(hr, start, end):
    hr = hr[hr["startDate"] >= start]
    hr = hr[hr["endDate"] <= end]
    return hr

last_workout = workout_data.iloc[[-workout_idx_from_end]]
heartrate_data = record_data[record_data["type"] == "HeartRate"]

# Extract heartrate statistics for certain workout
heartrate_workout = get_heartrate_for_date(heartrate_data, pd.to_datetime(1719407328, unit='s', utc=True),
                                           pd.to_datetime(1721480928, unit='s', utc=True))

# value is numeric, NaN if fails
heartrate_workout['value'] = pd.to_numeric(heartrate_workout['value'], errors='coerce')

# some records do not measure anything, just count occurences
# filling with 1.0 (= one time) makes it easier to aggregate
heartrate_workout['value'] = heartrate_workout['value'].fillna(1.0)

print(heartrate_workout.head())

heartrate_workout.to_csv('hr.csv', index=False)
