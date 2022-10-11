from scipy import signal
import numpy as np
name_indicies = ['energy_20_30KHz', 'totot']

def compute_ecoacoustics(wav, sr, ref_dB, Fmin, Fmax):
    # Filter
    b, a = signal.butter(2, (20*1e3, 30*1e3) ,fs=sr, btype='band')
    wav = signal.filtfilt(b, a, wav)
    energy_20_30KHz = np.max(np.abs(wav))/np.std(wav)

    
    return({'energy_20_30KHz': energy_20_30KHz, 'totot' : np.random.rand(1)})