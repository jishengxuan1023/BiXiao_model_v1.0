import numpy as np
import os
import matplotlib.pyplot as plt


# Load forecast data (a)
a = np.load('output_env/env_fc_2023102900.npy')  # (12, 6, 30)

# Load observation data (b)

b_list = []
for file in sorted(os.listdir('envData_for_plots/')):
    b_list.append(np.load('envData_for_plots/' + file))
b = np.array(b_list)   # (12, 6, 30)


# Basic info
num_times = a.shape[0]
num_vars  = a.shape[1]
num_sites = a.shape[2]
var_names = ['PM2.5','PM10','SO2','NO2','O3','CO']

# y-axis ranges for each pollutant (adjust as needed)
y_ranges = {
    'PM2.5': (0, 200),
    'PM10':  (0, 300),
    'O3':    (0, 250),
    'SO2':   (0, 80),
    'NO2':   (0, 150),
    'CO':    (0, 5)
}


# Remove site index = 8 and compute spatial mean
valid_inds = [i for i in range(num_sites) if i != 8]
a_mean = a[:, :, valid_inds].mean(axis=2)   # (12, 6)
b_mean = b[:, :, valid_inds].mean(axis=2)   # (12, 6)

# Time axis as simple index
t = np.arange(num_times)

# Plotting: 6 subplots (one per pollutant)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for var_idx in range(num_vars):
    ax = axs[var_idx]
    var_name = var_names[var_idx]
    y_min, y_max = y_ranges[var_name]

    y_new = a_mean[:, var_idx]
    y_obs = b_mean[:, var_idx]

    # Plot lines
    ax.plot(t, y_new, label='BiXiao_Forecasts', linewidth=2)
    ax.plot(t, y_obs, label='Observations', linewidth=2)

    # Set titles and axes
    ax.set_title(var_name)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Valid Time Steps")
    ax.set_ylabel(var_name)
    ax.grid(True)

    # Legend inside each subplot
    ax.legend(loc='upper right', fontsize=9)

# Main title
plt.suptitle("Spatial-mean Forecast vs Observation", fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save figure
plt.savefig("plot.png", dpi=300)
# plt.show()

print("Saved as plot.png")
