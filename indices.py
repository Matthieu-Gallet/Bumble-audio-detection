from scipy import signal
import numpy as np

from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI, compute_spectrogram
from utils.alpha_indices import acoustic_events, acoustic_activity, spectral_entropy, bioacousticsIndex
from scipy.signal import butter, filtfilt
from utils import OctaveBand


name_indicies = ['energy_band1','energy_band2','dB']



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def compute_ecoacoustics(wav, sr, ref_dB, Fmin, Fmax):
    # Filter
    #b, a = signal.butter(2, (Fmin, Fmax) ,fs=sr, btype='band')
    #wav = signal.filtfilt(b, a, wav)
    #energy_10_20KHz = np.max(np.abs(wav))/np.std(wav)

    #b, a = signal.butter(2, (15000, 28500) ,fs=sr, btype='band')
    #wav_1 = signal.filtfilt(b, a, wav)
    #energy_band1 = np.max(np.abs(wav_1))/np.std(wav_1)

    #b, a = signal.butter(2, (35000, Fmax) ,fs=sr, btype='band')
    #wav_2 = signal.filtfilt(b, a, wav)
    #energy_band2 = np.max(np.abs(wav_2))/np.std(wav_2)

    #wavforme = butter_bandpass_filter(wav, Fmin, Fmax, fs=sr)
    #dB_band, _ = OctaveBand.octavefilter(wavforme, fs=sr, fraction=1, order=4, limits=[100, 20000], show=0)
    # dB_band = [0]*8
    
    dB = 20 * np.log10(np.std(wav))
    

    return({'dB': dB})