"""AutoMe Wearable Project - Parkinson's Tremor Signal Processing.

An IMU (inertial measurement unit) reports acceleration and rotation.
This script uses the X-axis acceleration to look for tremor-like motion.

Processes 6D IMU data (MPU6050 / PADS text files) without header lines.
"""

# NumPy provides fast arrays and mathematical operations on the sensor samples.
import numpy as np
# pandas reads the sensor text file into a labeled table (DataFrame).
import pandas as pd
# Matplotlib creates and saves the graphs that help us inspect the signal.
import matplotlib.pyplot as plt
# SciPy provides digital-filter and Fourier-transform tools.
from scipy import signal
from scipy.fft import fft, fftfreq


# ---------------------------------------------------------------------------
# Section 1: Load the sensor data
# ---------------------------------------------------------------------------
# This function is responsible only for reading the file and labeling its
# columns. Keeping file input separate makes the later signal-processing code
# easier to read and reuse with another dataset.
# "def" starts a function definition. A function is a named, reusable group
# of instructions. file_path is an input parameter supplied by the caller.
def load_imu_data(file_path):
    """
    Loads IMU CSV/TXT datasets that lack header names.
    Assigns standard column names for 7 comma-separated values:
    [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    """
    # The input file has seven comma-separated values per row. A Python list
    # uses square brackets and stores several values in one variable.
    # These names make each column easier to understand later.
    column_names = [
        'timestamp_sec', 
        'accel_x_g', 
        'accel_y_g', 
        'accel_z_g', 
        'gyro_x', 
        'gyro_y', 
        'gyro_z'
    ]
    
    # header=None means the file does not contain a title row.
    # names=column_names gives the columns useful labels.
    # The result is stored in df, a pandas DataFrame (a spreadsheet-like
    # table). The "=" sign assigns the value on the right to the name on left.
    df = pd.read_csv(file_path, header=None, names=column_names)
    # len(df) counts the rows that were loaded.
    print(f"Successfully loaded {len(df)} rows from '{file_path}'.")
    return df


# ---------------------------------------------------------------------------
# Section 2: Remove unwanted frequencies
# ---------------------------------------------------------------------------
# A signal is made from many frequencies. This function keeps the range that
# may contain tremor and removes very slow movement and high-frequency noise.
def apply_bandpass_filter(raw_signal, sampling_rate, lowcut=1.0, highcut=12.0, order=4):
    """
    Applies a Butterworth Bandpass Filter (1 Hz - 12 Hz).
    Removes baseline gravity offset (<1 Hz) and high-frequency noise (>12 Hz).
    """
    # The Nyquist frequency is half the sampling rate. It is the highest
    # frequency that can be represented reliably by this sampled signal.
    # Digital filters must use cutoff frequencies below this value.
    nyquist = 0.5 * sampling_rate
    # Convert the cutoff frequencies to the normalized values
    # expected by scipy's Butterworth filter.
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create the filter's coefficients. The filter keeps frequencies between
    # lowcut and highcut and reduces frequencies outside that range.
    # b and a receive two values at once. Python can unpack a returned pair
    # into two variables in one assignment.
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    # Apply the filter forward and backward. This removes phase shift, so a
    # movement remains aligned with its original timestamp.
    filtered_signal = signal.filtfilt(b, a, raw_signal)
    return filtered_signal


# ---------------------------------------------------------------------------
# Section 3: Find important frequencies with an FFT
# ---------------------------------------------------------------------------
# The FFT (Fast Fourier Transform) helps answer: "Which frequencies are
# strongest in this movement?" This is useful because tremor is frequency-based.
def compute_fft_spectrum(filtered_signal, sampling_rate):
    """
    Computes Fast Fourier Transform (FFT) to detect 3–7 Hz tremor peaks.
    """
    # N is the number of measurements in the signal. len() returns the number
    # of items in a list or array.
    N = len(filtered_signal)
    # dt is the time between two measurements.
    dt = 1.0 / sampling_rate
    
    # The raw sensor values describe movement over time (time domain). The FFT
    # rewrites that same information as the strength of each frequency.
    yf = fft(filtered_signal)
    # Create the frequency value belonging to each FFT result.
    # Keep only the positive half of the frequencies. [:N // 2] is slicing:
    # it selects items from the beginning up to, but not including, this index.
    xf = fftfreq(N, dt)[:N // 2]
    # FFT results are complex numbers. Their absolute values are magnitudes,
    # which tell us how strongly each frequency appears in the signal.
    magnitude = 2.0 / N * np.abs(yf[0:N // 2])
    
    # Create a Boolean mask: True for values inside 3-7 Hz, False otherwise.
    # NumPy uses this mask to select matching array elements.
    band_mask = (xf >= 3.0) & (xf <= 7.0)
    freq_in_band = xf[band_mask]
    mag_in_band = magnitude[band_mask]
    
    # "if" chooses code based on a condition. This checks that the band
    # contains at least one frequency before trying to find its maximum.
    if len(mag_in_band) > 0:
        # Find the largest magnitude in the 3–7 Hz tremor range.
        # np.argmax returns the position (index), not the magnitude itself.
        peak_index = np.argmax(mag_in_band)
        dominant_freq = freq_in_band[peak_index]
        peak_amplitude = mag_in_band[peak_index]
    else:
        # Use zero values when the requested frequency range is empty.
        dominant_freq = 0.0
        peak_amplitude = 0.0
        
    # A function can return several values separated by commas. The caller
    # can unpack them into separate variables.
    return xf, magnitude, dominant_freq, peak_amplitude


# ---------------------------------------------------------------------------
# Section 4: Draw and save the results
# ---------------------------------------------------------------------------
# Graphs make it easier to compare the original signal, cleaned signal, and
# frequency peak than reading numerical values alone.
def plot_tremor_analysis(time_sec, raw_signal, filtered_signal, frequencies, magnitude, dominant_freq, peak_amplitude):
    """
    Generates time-domain and frequency spectrum plots saved as an image.
    """
    # Make one figure with two views of the same data:
    # 1) movement over time and 2) signal strength by frequency.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    
    # Plot 1: Time domain - compare the original sensor values with the
    # filtered values to see what noise was removed.
    # Each method call below changes the first plot (ax1). A dot means "call
    # this method belonging to this object."
    ax1.plot(time_sec, raw_signal, label='Raw Accelerometer Signal (Noise + Gravity)', color='lightgray', linewidth=1)
    ax1.plot(time_sec, filtered_signal, label='Filtered Signal (1–12 Hz Bandpass)', color='#1f77b4', linewidth=1.8)
    ax1.set_title('AutoMe Wearable IMU Data: Time-Domain Tremor Signal', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Frequency domain - show the full FFT and highlight the
    # 3-7 Hz search band and its strongest frequency.
    ax2.plot(frequencies, magnitude, color='#d62728', linewidth=1.8, label='FFT Spectrum')
    ax2.axvspan(3.0, 7.0, color='orange', alpha=0.2, label='Parkinson Tremor Band (3–7 Hz)')
    ax2.axvline(dominant_freq, color='darkred', linestyle='--', linewidth=1.5,
                label=f'Peak: {dominant_freq:.2f} Hz ({peak_amplitude:.3f} g)')
    ax2.set_title('AutoMe Wearable IMU Data: Frequency Spectrum (FFT Analysis)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (g)')
    ax2.set_xlim(0, 15)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust spacing so labels do not overlap, then save the image.
    # dpi controls image resolution. close() releases the figure from memory.
    plt.tight_layout()
    plt.savefig('tremor_analysis_plot.png', dpi=150)
    plt.close()
    print("Graph saved as 'tremor_analysis_plot.png'.")

def main():
    # -----------------------------------------------------------------------
    # Section 5: Coordinate the complete analysis
    # -----------------------------------------------------------------------
    # main() is the workflow controller: load data, estimate timing, filter
    # the signal, find its frequency peak, print a summary, and save a plot.
    # This is the file that contains the sensor measurements. A string is text
    # surrounded by quotes.
    data_file = '001_CrossArms_LeftWrist.txt'
    
    # 1. Load data
    # Call the loading function and store the returned table in df.
    df = load_imu_data(data_file)
    # .values converts pandas columns into NumPy arrays for calculations.
    # The two assignments select the timestamp and X-axis columns we need.
    time_sec = df['timestamp_sec'].values
    raw_accel_x = df['accel_x_g'].values
    
    # Determine the sampling rate from timestamps. The sampling rate tells
    # signal-processing functions how many measurements were taken per second.
    # At least two timestamps are needed to measure a time interval.
    # The indented block belongs to the if statement. Python uses indentation
    # instead of braces to show which lines belong together.
    if len(time_sec) > 1:
        # np.diff finds neighboring time differences.
        # The median is one robust representative interval.
        sample_interval = float(np.median(np.diff(time_sec)))
        # Frequency means "how many samples per second".
        # This is a conditional expression: use the calculated rate when the
        # interval is valid; otherwise use 100.0.
        sampling_rate = 1.0 / sample_interval if sample_interval > 0 else 100.0
    else:
        # Use 100 Hz as a safe fallback when timestamps are unavailable.
        sampling_rate = 100.0
    
    # 2. Filter noise and slow gravity changes from the X-axis signal.
    filtered_signal = apply_bandpass_filter(raw_accel_x, sampling_rate)
    
    # 3. Convert the filtered signal to the frequency domain and search for
    # the strongest component in the tremor band.
    freqs, mag, dom_freq, peak_amp = compute_fft_spectrum(filtered_signal, sampling_rate)
    
    # 4. Print the measurements and apply a simple screening rule.
    print("\n==========================================")
    print("      AUTOME CLINICAL TREMOR ANALYSIS     ")
    print("==========================================")
    print(f"Sampling Rate      : {sampling_rate:.1f} Hz")
    print(f"Dominant Frequency : {dom_freq:.2f} Hz")
    print(f"Peak Tremor Power  : {peak_amp:.4f} g")
    
    # "and" requires both conditions to be true. This is a simple threshold
    # rule, not a medical diagnosis.
    if 3.0 <= dom_freq <= 7.0 and peak_amp > 0.05:
        print("Clinical Status    : POSITIVE for Parkinsonian Tremor (3–7 Hz)")
    else:
        print("Clinical Status    : NEGATIVE / No Tremor Detected")
    print("==========================================\n")
    
    # 5. Save a visual report of both the time and frequency analysis.
    # Create and save the time and frequency plots.
    plot_tremor_analysis(time_sec, raw_accel_x, filtered_signal, freqs, mag, dom_freq, peak_amp)

# Run main() only when this file is started directly, not when imported.
# Python sets __name__ to '__main__' for the file being run. This guard lets
# another file import these functions without automatically running the whole
# analysis.
if __name__ == '__main__':
    main()
