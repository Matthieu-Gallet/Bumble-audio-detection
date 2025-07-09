import os
from utils import utils


from tqdm import tqdm
import librosa
import torchaudio
import pandas as pd
import numpy as np


def get_file_list(path_audio_folder):
    """return list of wav file in a give folder"""

    wav_files = []
    for root, dirs, files in os.walk(path_audio_folder, topdown=False):
        for name in files:
            if name[-3:].casefold() == "wav" and name[:2] != "._":
                wav_files.append(os.path.join(root, name))

    return wav_files


def metadata_generator(folder, file_format):
    """Generate meta data for one folder (one site) and save in csv and pkl"""

    filelist = []
    Df = pd.DataFrame(columns=["filename", "datetime", "length", "sr"])
    Df_error = pd.DataFrame(columns=["filename"])
    if file_format == "wav":
        n = 3
    elif file_format == "flac":
        n = 4
    else:
        raise (f"Format: {file_format} is not allowed")

    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name[-n:].casefold() == file_format and name[:2] != "._":
                filelist.append(os.path.join(root, name))

    for idx, wavfile in enumerate(tqdm(filelist)):
        _, meta = utils.read_audio_hdr(
            wavfile, False, file_format=file_format
        )  # meta data
        x, sr = librosa.load(wavfile, sr=None, mono=True)
        try:
            x, sr = librosa.load(wavfile, sr=None, mono=True)
        except:
            print("skipping short file")
            continue

        Df = pd.concat(
            [
                Df,
                pd.DataFrame(
                    {
                        "datetime": [meta["datetime"]],
                        "filename": [wavfile],
                        "length": [len(x)],
                        "sr": [sr],
                        "dB": 10 * np.log10(np.std(x) ** 2),
                    }
                ),
            ],
            ignore_index=True,
        )
    Df = Df.sort_values("datetime").reset_index()
    return Df
