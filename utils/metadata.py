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

    # Collect all dataframes in a list instead of concatenating in loop
    dataframes_list = []
    corrupted_files = []

    for idx, wavfile in enumerate(tqdm(filelist)):
        try:
            _, meta = utils.read_audio_hdr(
                wavfile, False, file_format=file_format
            )  # meta data

            # Try to load the audio file
            x, sr = librosa.load(wavfile, sr=None, mono=True)

            # Create individual dataframe and add to list
            df_row = pd.DataFrame(
                {
                    "datetime": [meta["datetime"]],
                    "filename": [wavfile],
                    "length": [len(x)],
                    "sr": [sr],
                    "dB": 10 * np.log10(np.std(x) ** 2),
                }
            )
            dataframes_list.append(df_row)

        except Exception as e:
            print(f"Skipping corrupted file: {wavfile} - Error: {str(e)}")
            corrupted_files.append(wavfile)
            continue

    # Concatenate all dataframes at once (more efficient and no warning)
    if dataframes_list:
        Df = pd.concat(dataframes_list, ignore_index=True)
    else:
        Df = pd.DataFrame(columns=["filename", "datetime", "length", "sr", "dB"])

    # Save corrupted files list if any
    if corrupted_files:
        corrupted_file_path = os.path.join(folder, "corrupted_files.txt")
        with open(corrupted_file_path, "w") as f:
            f.write("# Liste des fichiers corrompus ou illisibles\n")
            f.write(f"# Générée le: {pd.Timestamp.now()}\n")
            f.write(f"# Nombre de fichiers corrompus: {len(corrupted_files)}\n\n")
            for corrupt_file in corrupted_files:
                f.write(f"{corrupt_file}\n")
        print(
            f"⚠️  {len(corrupted_files)} fichiers corrompus trouvés et listés dans: {corrupted_file_path}"
        )
    else:
        print("✅ Aucun fichier corrompu détecté")

    Df = Df.sort_values("datetime").reset_index()
    return Df
