import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack

' Fourier Transform '
' Time Domain -> Frequency Domain'
def fft(t_values, sampling_rate, return_xf=False, makeplot=False): # t_values = input data in time domain, sampling_rate in Hz
    # FFT
    f_values = fftpack.fft(t_values)
    # Normalize the output to get amplitude equalt to the original input data
    number_of_data_points = len(f_values)
    positive_spectrum = f_values[:len(f_values)//2] # // = floor division
    f_values_norm = 2.0/number_of_data_points * np.abs(positive_spectrum)
    
    if makeplot is True:
        xf = np.linspace(0.0, int(sampling_rate/(2.0)), int(number_of_data_points/2)) # FFT results in frequency ranges from 0 to sampling_rate/2. Number of frequencies = half of number of the number of samples
        plt.plot(xf, f_values_norm, label="FFT results (+ve spectrum with normalized amplitude)", color="olive", linewidth=1)
        plt.legend(loc="upper right")
        plt.xlabel("Freq")
        plt.ylabel("Amplitude (arbitrary unit)")
        # plt.ylim(bottom=-1,top=10+np.average(f_values_norm))
        # plt.ylim(bottom=0,top=np.average(f_values_norm))
        return f_values_norm, xf
    
    if return_xf:
        xf = np.linspace(0.0, int(sampling_rate/(2.0)), int(number_of_data_points/2)) # FFT results in frequency ranges from 0 to sampling_rate/2. Number of frequencies = half of number of the number of samples
        return f_values_norm, xf
    return f_values_norm

' Frquency Band filtering using butterworth filter '
def bandpass_filter(t_values, sampling_rate, cutoff_freq=[5,15], filter="butterworth"):  # t_values = input data in time domain, sampling_rate in Hz
    if filter == "butterworth":
        if len(cutoff_freq) != 2 : 
            print("ERROR : Band pasàs filter accepts cutoff_freq=[LF, HF]")
            print("if you need a low pass filter, cutoff in range [0,cutoff_freq_high]")
            print("if you need a high pass filter, cutoff in range [cutoff_freq_low,cutoff_freq_low]")
            pass
        # Create a band pass butterworth filter 
        filter_w_low = cutoff_freq[0] / (sampling_rate / 2) # Normalize the frequency
        filter_w_high = cutoff_freq[1] / (sampling_rate / 2) # Normalize the frequency
        if filter_w_low == 0:
            filter_b, filter_a = signal.butter(4, filter_w_high, 'low')
        elif filter_w_low == filter_w_high:
            filter_b, filter_a = signal.butter(4, filter_w_low, "highpass")
        else : filter_b, filter_a = signal.butter(4, [filter_w_low,filter_w_high], 'bandpass')

        # Filter output in time domain
        filter_t_values = signal.filtfilt(filter_b, filter_a, t_values)
        return filter_t_values
    
    else :
        print("Only butterworth filter is implemented for the moment")
        pass
    