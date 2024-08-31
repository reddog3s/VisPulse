import pandas as pd
import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import argparse

def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))

parser = argparse.ArgumentParser(description='Project')
parser.add_argument('--method', required=True,
                    help='method')

if __name__ == "__main__":
    args = parser.parse_args()
    method = args.method
else:
    method = 'POS'
vid_size = (1920,1080)

hr_mean = []
hr_std = []
ids = []
vid_ids = []

hr_real_mean = []
hr_real_std= []
ids_real = []
vid_ids_real = []

vid_csv_path = os.path.join('/mnt','d', 'test', 'iphone','vid_data.csv')
base_path_pred = os.path.join('./results','hr_results','predicted',method)
base_path_gt = os.path.join('./results','hr_results','gt')
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

for vid_id in vid_data['vid_id']:
    curr_vid_data = vid_data[vid_data['vid_id'] == vid_id]
    vid_id = curr_vid_data['vid_id'].iloc[0]
    watches = [curr_vid_data['watch_right'].iloc[0], curr_vid_data['watch_left'].iloc[0]]
    people = [curr_vid_data['person_right'].iloc[0], curr_vid_data['person_left'].iloc[0]]

    for watch_id, person_id in zip(watches, people):
        if watch_id >= 0:
            gt_csv_file_name = 'hr_vid_' + str(vid_id) + '_watch_' + str(watch_id) + '.csv'
            csv_path_gt = os.path.join(base_path_gt, gt_csv_file_name)
            df_gt = pd.read_csv(csv_path_gt)
            hr_real_mean.append(df_gt['value'].mean())
            hr_real_std.append(df_gt['value'].std())
            ids_real.append(person_id)
            vid_ids_real.append(vid_id)
    

    csv_path = os.path.join(base_path_pred, 'hr_' + str(vid_id) + '.csv')
    df = pd.read_csv(csv_path)

    person_ids = df['person_id'].unique()
    num_person = len(person_ids)
    for id in person_ids:
        person_df = df[df['person_id'] == id]
        person_df = df[df['hr'] > 0]

        if id == 0:
            id = 1
        elif id == 1:
            id = 0

        hr_mean.append(person_df['hr'].mean())
        hr_std.append(person_df['hr'].std())
        ids.append(id)
        vid_ids.append(vid_id)
    

results_gt = pd.DataFrame({
    "vid_id_gt": vid_ids_real,
    "person_id_gt": ids_real,
    "hr_mean_gt": hr_real_mean,
    "hr_std_gt": hr_real_std
})
results_gt = results_gt.sort_values(by=['vid_id_gt', 'person_id_gt'])
results = pd.DataFrame({
    "vid_id": vid_ids,
    "person_id": ids,
    "hr_mean": hr_mean,
    "hr_std": hr_std
})
results = results.sort_values(by=['vid_id', 'person_id'])
results_all = results.join(results_gt)
print(results_all)

aprox = results_all['hr_mean']
real = results_all['hr_mean_gt']

# mean value for every vid 

# to lists / pandas

# rest for these mean values

print('RMSE: ', rmse(aprox, real))

mean_real = np.mean(real)
mean_aprox = np.mean(aprox)

print("\nReal mean = %.3f, Real std = %.3f" % (mean_real, np.std(real)))
print("Aprox mean = %.3f, Aprox std = %.3f" % (mean_aprox, np.std(aprox)))

mean_err = np.mean(abs(real - aprox))
print("Mean absolute error = %.3f, Mean relative error = %.3f %%" % (mean_err, (mean_err/mean_real)*100))
print(st.pearsonr(aprox, real))



### Bland-Altman analysis
### based on:
### https://rowannicholls.github.io/python/statistics/agreement/bland_altman.html

means = (aprox + real) / 2
diffs = aprox - real


# Average difference (aka the bias)
bias = np.mean(diffs)
# Sample standard deviation
s = np.std(diffs, ddof=1)  # Use ddof=1 to get the sample standard deviation

print(f'For the differences, x̄ = {bias:.2f} m/s and s = {s:.2f} m/s')

# Limits of agreement (LOAs)
upper_loa = bias + 1.96 * s
lower_loa = bias - 1.96 * s

print(f'The limits of agreement are {upper_loa:.2f} m/s and {lower_loa:.2f} m/s')


# Confidence level
C = 0.95  # 95%
# Significance level, α
alpha = 1 - C
# Number of tails
tails = 2
# Quantile (the cumulative probability)
q = 1 - (alpha / tails)
# Critical z-score, calculated using the percent-point function (aka the
# quantile function) of the normal distribution
z_star = st.norm.ppf(q)

print(f'95% of normally distributed data lies within {z_star}σ of the mean')

# Limits of agreement (LOAs)
loas = (bias - z_star * s, bias + z_star * s)

print(f'The limits of agreement are {loas} m/s')

# Limits of agreement (LOAs)
loas = st.norm.interval(C, bias, s)

print(np.round(loas, 2))


# Create plot
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=loas[1], c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=loas[0], c='grey', ls='--')
# Labels
ax.set_title('Bland-Altman Plot')
ax.set_xlabel('Mean (m/s)')
ax.set_ylabel('Difference (m/s)')
# Get axis limits
left, right = ax.get_xlim()
bottom, top = ax.get_ylim()
# Set y-axis limits
max_y = max(abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left, left + domain * 1.1)
# Annotations
ax.annotate('+LoA', (right, upper_loa), (0, 7), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels')
ax.annotate('-LoA', (right, lower_loa), (0, 7), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels')
# Show plot
plt.show()