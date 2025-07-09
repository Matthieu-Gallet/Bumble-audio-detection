import numpy as np

def compute_ecoacoustics(wav):
    
    dB = 20 * np.log10(np.std(wav))
    
    return({'dB': dB})