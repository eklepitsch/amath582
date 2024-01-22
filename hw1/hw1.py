import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_path = r'C:\Users\eklep\Google Drive\grad school\amath582\homework\\' + \
            r'hw1\subdata.npy'
d = np.load(data_path)

# Define the dimensions of our time domain data
N_grid = 64
N_measurements = 49
grid_shape = (N_grid, N_grid, N_grid)

# Preallocate an array to store the FFT of each measurement
results = np.empty((N_measurements, N_grid, N_grid, N_grid),
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
k = np.arange(0, N_grid)
fig_noise, ax_noise = plt.subplots(1, 3)
ax_noise[0].plot(k, f_hat_avg[:, center_frequency_bin[1],
                 center_frequency_bin[2]])
ax_noise[0].axvline(center_frequency_bin[0], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[0]}')
ax_noise[0].set_title(r'Center frequency in $\bf{x}$-direction')
ax_noise[0].set_xlabel('Frequency bin (k)')
ax_noise[0].set_ylabel('Magnitude')
ax_noise[0].legend()
ax_noise[1].plot(k, f_hat_avg[center_frequency_bin[0], :,
                 center_frequency_bin[2]])
ax_noise[1].axvline(center_frequency_bin[1], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[1]}')
ax_noise[1].set_title(r'Center frequency in $\bf{y}$-direction')
ax_noise[1].set_xlabel('Frequency bin (k)')
ax_noise[1].set_ylabel('Magnitude')
ax_noise[1].legend()
ax_noise[2].plot(k, f_hat_avg[center_frequency_bin[0],
                 center_frequency_bin[1], :])
ax_noise[2].axvline(center_frequency_bin[2], color='r', linestyle='--',
                    label=f'Center frequency = {center_frequency_bin[2]}')
ax_noise[2].set_title(r'Center frequency in $\bf{z}$-direction')
ax_noise[2].set_xlabel('Frequency bin (k)')
ax_noise[2].set_ylabel('Magnitude')
ax_noise[2].legend()
fig_noise.set_figwidth(15)
fig_noise.tight_layout(pad=3)
fig_noise.show()
