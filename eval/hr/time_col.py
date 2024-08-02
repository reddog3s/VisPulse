import os
import pandas as pd

csv_dir_path = os.path.join('/home','wojtek', 'magisterka', 'results','hr_results','gt')
vid_csv_path = os.path.join('/mnt','d', 'test', 'iphone','vid_data.csv')

watches = ['watch_right', 'watch_left']

vid_data = pd.read_csv(vid_csv_path, date_format='%Y-%m-%d %H:%M:%S%z',
                             parse_dates=['vid_start','vid_end'],
                             dtype={
                                'vid_id': 'int',
                                'vid_start': 'string',
                                'vid_end': 'string',
                                'person_left': 'string',
                                'watch_left': 'int',
                                'person_right': 'string',
                                'watch_right': 'int',
                                'exercise': 'string',
                                'file_path': 'string'
                            })

for watch_id in range(2):
    watch_name = watches[watch_id]
    curr_watch_vids = vid_data[vid_data[watch_name] == watch_id]

    for vid_id in curr_watch_vids['vid_id']:
        video_start = curr_watch_vids[curr_watch_vids['vid_id'] == vid_id]['vid_start'].iloc[0]
        video_end = curr_watch_vids[curr_watch_vids['vid_id'] == vid_id]['vid_end'].iloc[0]

        csv_path = os.path.join(csv_dir_path, 'hr_vid_' + str(vid_id) + '_watch_' + str(watch_id) + '.csv')

        heartrate_data = pd.read_csv(csv_path, date_format='%Y-%m-%d %H:%M:%S%z',
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
        # filter results taken after video end
        heartrate_data = heartrate_data[heartrate_data['endDate'] < video_end]
        heartrate_data = heartrate_data[heartrate_data['startDate'] > video_start]

        dates = heartrate_data['endDate']
        vid_time = (dates - video_start).dt.total_seconds()
        heartrate_data['videoTime'] = vid_time
        #csv_path_save = os.path.join(csv_dir_path, 'hr_vid_' + str(vid_id) + '_watch_' + str(watch_id) + '_edit.csv')
        heartrate_data.to_csv(csv_path, index=False)

