from scipy import signal
import numpy as np

from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI, compute_spectrogram
from utils.alpha_indices import acoustic_events, acoustic_activity, spectral_entropy, bioacousticsIndex
from scipy.signal import butter, filtfilt
from utils import OctaveBand


name_indicies = ['energy_10_20KHz','dB', 'ndsi_N', 'aci_N' ,
                    'BI_N' , 'EAS_N' ,
                    'ECV_N' , 'EPS_N'  ,'ndsi_W' , 'aci_W' ,
                    'BI_W' , 'EAS_W' ,
                    'ECV_W' , 'EPS_W' , 'ACT' ,
                    'POWERB_126' , 'POWERB_251' , 'POWERB_501' , 
                    'POWERB_1k' , 'POWERB_2k' , 'POWERB_4k' , 
                    'POWERB_8k' , 'POWERB_16k' ]



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
    b, a = signal.butter(2, (Fmin, Fmax) ,fs=sr, btype='band')
    wav = signal.filtfilt(b, a, wav)
    energy_10_20KHz = np.max(np.abs(wav))/np.std(wav)

    wavforme = butter_bandpass_filter(wav, Fmin, Fmax, fs=sr)
    dB_band, _ = OctaveBand.octavefilter(wavforme, fs=sr, fraction=1, order=4, limits=[100, 20000], show=0)
    # dB_band = [0]*8
    Sxx, freqs = compute_spectrogram(wavforme, sr)

    # Sxx_dB = 10 * np.log10(Sxx)
    # N = len(wavforme)
    dB = 20 * np.log10(np.std(wavforme))
    # nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband=200, normalization=True, slopes=(1 / 75, 1 / 75))
    # wide : 2000 - 20000 : narrow : 5000 : 15000
    min_anthro_bin = np.argmin([abs(e - 5000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 15000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_N, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin,:], freqs[min_anthro_bin:max_anthro_bin], 100, sr) 
    ndsi_N = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 5000], biophony=[5000, 15000])
    bi_N = bioacousticsIndex(Sxx, freqs, frange=(5000, 15000), R_compatible=False)

    min_anthro_bin = np.argmin([abs(e - 2000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 20000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_W, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin, :], freqs, 100, sr)  
    ndsi_W = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 2000], biophony=[2000, 20000])
    bi_W = bioacousticsIndex(Sxx, freqs, frange=(2000, 20000), R_compatible=False)

    # wide 1000 - 20000 narrow 5000 - 15000


    EAS_N, _, ECV_N, EPS_N, _, _ = spectral_entropy(Sxx, freqs, frange=(5000, 15000))
    EAS_W, _, ECV_W, EPS_W, _, _ = spectral_entropy(Sxx, freqs, frange=(2000, 20000))


    ### specific code to calculate ACT and EVN with many ref Db offset

    _, ACT, _ = acoustic_activity(10*np.log10(np.abs(wavforme)**2), dB_threshold=ref_dB + 12, axis=-1)
    ACT = np.sum(np.asarray(ACT))/sr


    return({'energy_10_20KHz': energy_10_20KHz, 'dB': dB, 'ndsi_N': ndsi_N, 'aci_N': aci_N,
                  'BI_N': bi_N, 'EAS_N': EAS_N,
                  'ECV_N': ECV_N, 'EPS_N': EPS_N,'ndsi_W': ndsi_W, 'aci_W': aci_W,
                  'BI_W': bi_W, 'EAS_W': EAS_W,
                  'ECV_W': ECV_W, 'EPS_W': EPS_W, 'ACT':ACT,
                  'POWERB_126':dB_band[0], 'POWERB_251':dB_band[1], 
                  'POWERB_501':dB_band[2], 'POWERB_1k':dB_band[3], 
                  'POWERB_2k':dB_band[4], 'POWERB_4k':dB_band[5], 
                  'POWERB_8k':dB_band[6], 'POWERB_16k':dB_band[7]})