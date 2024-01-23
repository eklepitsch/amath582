import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load the data
data_path = r'C:\Users\eklep\Google Drive\grad school\amath582\homework\\' + \
            r'hw1\subdata.npy'
d = np.load(data_path)

# Define the dimensions of our time domain data
N_grid = 64
N_measurements = 49
grid_shape = (N_grid, N_grid, N_grid)
t = np.arange(0, N_measurements)
t_hours = t * 0.5

# Define the dimensions of our frequency domain data
k = np.arange(0, N_grid)

# Preallocate an array to store the FFT of each measurement
results = np.empty((N_measurements, N_grid, N_grid, N_grid),
                   dtype=np.complex_)

# Preallocate an array to store the filtered frequency domain data
results_filtered = np.empty((N_measurements, N_grid, N_grid, N_grid),
                            dtype=np.complex_)

# Preallocate an array to store the reconstructed (filtered) time domain data
time_signal_filtered = np.empty((N_measurements, N_grid, N_grid, N_grid),
                                dtype=np.complex_)

# Take the FFT of each measurement
for i in range(0, N_measurements):
    measurements = np.reshape(d[:, i], (N_grid, N_grid, N_grid))
    f_hat = np.fft.fftshift(np.fft.fftn(measurements))
    results[i, :, :, :] = f_hat

# Average the FFTs
f_hat_avg = abs(np.average(results, axis=0))

# Find the center frequency of the submarine.  The center frequency corresponds
# to the highest weighted frequency bin in the averaged FFT.
center_frequency_bin = np.unravel_index(np.argmax(f_hat_avg), grid_shape)
center_frequency_bin_value = f_hat_avg[center_frequency_bin]
print(f'Center frequency bin: {center_frequency_bin}')
print(f'Value in center frequency bin: {center_frequency_bin_value}')

# Make some plots of the averaged FFT in each direction, at the point of the
# center frequency.  This will show us the noise profile in each direction.
fig_noise, ax_noise = plt.subplots(1, 3)
ax_noise[0].plot(k, f_hat_avg[:, center_frequency_bin[1],
                 center_frequency_bin[2]], linewidth=3)
ax_noise[0].axvline(center_frequency_bin[0], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[0]}')
ax_noise[0].set_title(r'$\bf{x}$-direction')
ax_noise[0].set_xlabel('Frequency bin (k)')
ax_noise[0].set_ylabel('Magnitude')
ax_noise[0].legend()
ax_noise[1].plot(k, f_hat_avg[center_frequency_bin[0], :,
                 center_frequency_bin[2]], linewidth=3)
ax_noise[1].axvline(center_frequency_bin[1], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[1]}')
ax_noise[1].set_title(r'$\bf{y}$-direction')
ax_noise[1].set_xlabel('Frequency bin (k)')
ax_noise[1].set_ylabel('Magnitude')
ax_noise[1].legend()
ax_noise[2].plot(k, f_hat_avg[center_frequency_bin[0],
                 center_frequency_bin[1], :], linewidth=3)
ax_noise[2].axvline(center_frequency_bin[2], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[2]}')
ax_noise[2].set_title(r'$\bf{z}$-direction')
ax_noise[2].set_xlabel('Frequency bin (k)')
ax_noise[2].set_ylabel('Magnitude')
ax_noise[2].legend()
fig_noise.set_figwidth(15)
fig_noise.tight_layout(pad=3)
fig_noise.suptitle('Time averaged FFT centered at the submarine\'s frequency')
fig_noise.show()


# Define a Gaussian filter
def gaussian(x, center=0, sigma=1, scale=1):
    return scale * np.exp(-(x - center)**2/sigma**2)


# Make some plots to show the filter design in each direction
fig_filter, ax_filter = plt.subplots(1, 3)
ax_filter[0].plot(k, f_hat_avg[:, center_frequency_bin[1],
                    center_frequency_bin[2]], linewidth=3)
ax_filter[0].set_title(r'$\bf{x}$-direction')
ax_filter[0].set_xlabel('Frequency bin (k)')
ax_filter[0].set_ylabel('Magnitude')
ax_filter[1].plot(k, f_hat_avg[center_frequency_bin[0], :,
                    center_frequency_bin[2]], linewidth=3)
ax_filter[1].set_title(r'$\bf{y}$-direction')
ax_filter[1].set_xlabel('Frequency bin (k)')
ax_filter[1].set_ylabel('Magnitude')
ax_filter[2].plot(k, f_hat_avg[center_frequency_bin[0],
                    center_frequency_bin[1], :], linewidth=3)
ax_filter[2].set_title(r'$\bf{z}$-direction')
ax_filter[2].set_xlabel('Frequency bin (k)')
ax_filter[2].set_ylabel('Magnitude')
fig_filter.set_figwidth(15)
fig_filter.tight_layout(pad=3)
fig_filter.suptitle('Gaussian filter design in each direction')

for sigma in range(1, 20):
    filter_x = gaussian(k, center_frequency_bin[0], sigma,
                        center_frequency_bin_value)
    filter_y = gaussian(k, center_frequency_bin[1], sigma,
                        center_frequency_bin_value)
    filter_z = gaussian(k, center_frequency_bin[2], sigma,
                        center_frequency_bin_value)
    if sigma in [1, 3, 10]:
        ax_filter[0].plot(k, filter_x, label=rf'$\sigma$ = {sigma}', alpha=0.75, linestyle='--')
    if sigma in [1, 5, 10]:
        ax_filter[1].plot(k, filter_y, label=rf'$\sigma$ = {sigma}', alpha=0.75, linestyle='--')
    if sigma in [1, 3, 10]:
        ax_filter[2].plot(k, filter_z, label=rf'$\sigma$ = {sigma}', alpha=0.75, linestyle='--')

ax_filter[0].legend()
ax_filter[1].legend()
ax_filter[2].legend()
fig_filter.show()

# In x direction, choose sigma = 3.
# In y direction, choose sigma = 5.
# in z direction, choose sigma = 3.
sigma_x = 3
sigma_y = 5
sigma_z = 3

# Create a 3-D Gaussian filter by multiplying 1-D filters
X, Y, Z = np.meshgrid(k, k, k, indexing='ij')
filter = gaussian(X, center_frequency_bin[0], sigma_x,
                  center_frequency_bin_value) * \
         gaussian(Y, center_frequency_bin[1], sigma_y,
                  center_frequency_bin_value) * \
         gaussian(Z, center_frequency_bin[2], sigma_z,
                  center_frequency_bin_value)

for i in range(0, N_measurements):
    # Apply the 3-D filter in the frequency domain
    results_filtered[i, :, :, :] = filter * results[i, :, :, :]

    # Convert back to the time domain
    time_signal_filtered[i, :, :, :] = np.fft.ifftn(
        np.fft.ifftshift(results_filtered[i, :, :, :]))

submarine_position = np.empty((N_measurements, 3))
for i in range(0, N_measurements):
    submarine_position[i] = np.array(np.unravel_index(
        np.argmax(time_signal_filtered[i, :, :, :]), grid_shape))

# Plot the submarine's path from the filtered time domain signal.
# First plot the path in each direction independently.
fig_position, ax_position = plt.subplots(1, 3)
ax_position[0].plot(t_hours, submarine_position[:, 0])
ax_position[0].set_title(r'$\bf{x}$-coordinate')
ax_position[0].set_xlabel('t (hours)')
ax_position[0].set_ylabel('position (grid coordinate)')
ax_position[1].plot(t_hours, submarine_position[:, 1])
ax_position[1].set_title(r'$\bf{y}$-coordinate')
ax_position[1].set_xlabel('t (hours)')
ax_position[1].set_ylabel('position (grid coordinate)')
ax_position[2].plot(t_hours, submarine_position[:, 2])
ax_position[2].set_title(r'$\bf{z}$-coordinate')
ax_position[2].set_xlabel('t (hours)')
ax_position[2].set_ylabel('position (grid coordinate)')
ax_position[2].yaxis.set_major_locator(MaxNLocator(integer=True))
fig_position.set_figwidth(15)
fig_position.tight_layout(pad=3)
fig_position.suptitle('Position of submarine')
fig_position.show()

# Now plot the 3-D path
fig_position_3d, ax_position_3d = plt.subplots(subplot_kw=dict(projection='3d'))
ax_position_3d.scatter3D(submarine_position[0, 0], submarine_position[0, 1],
                         submarine_position[0, 2], color='k', label='t = 0 hrs')  # First point
# Points 0 and 1 are the same, so skip point 1
ax_position_3d.scatter3D(submarine_position[2:-1, 0], submarine_position[2:-1, 1],
                         submarine_position[2:-1, 2])  # Middle points
ax_position_3d.scatter3D(submarine_position[-1, 0], submarine_position[-1, 1],
                         submarine_position[-1, 2], color='r', label='t = 24 hrs')  # Last point
ax_position_3d.view_init(30, 150)
ax_position_3d.set_xlabel('x')
ax_position_3d.set_ylabel('y')
ax_position_3d.set_zlabel('z')
ax_position_3d.legend()
fig_position_3d.suptitle('Position of submarine in 3-D')
fig_position_3d.show()

