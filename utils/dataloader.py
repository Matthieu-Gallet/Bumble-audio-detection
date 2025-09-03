import os
import torch
from torch.utils.data import Dataset

import librosa
import pandas as pd
from tqdm import tqdm
import datetime

import numpy as np
from scipy.signal import resample

from indices import compute_ecoacoustics
import torchaudio


class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader, len_audio_s, savepath, save_audio=True):
        self.meta = meta_dataloader
        self.sr_tagging = 32000
        self.savepath = savepath
        self.len_audio_s = len_audio_s
        self.save_audio = save_audio

        print(f"Dataset initialized with {len(self.meta)} audio segments")
        print(
            f"Segment length: {self.len_audio_s} seconds at Sample rate: {self.sr_tagging} Hz"
        )

    def __getitem__(self, idx):

        filename = self.meta["filename"][idx]

        # Load audio file, convert to mono, and apply offset and duration
        wav_o, sr = librosa.load(
            filename,
            sr=None,
            mono=True,
            offset=self.meta["start"][idx],
            duration=self.len_audio_s,
        )
        wav = torch.tensor(wav_o).view(1, len(wav_o))

        if self.save_audio:
            torchaudio.save(
                os.path.join(
                    self.savepath,
                    self.meta["date"][idx].strftime("%Y%m%d_%H%M%S") + ".flac",
                ),
                wav,
                sr,
                format="flac",
            )

        # Compute ecoacoustic indices
        ecoac = compute_ecoacoustics(wav_o)

        # Resample to audio tagging model sample rate
        wav = resample(wav_o, int(self.len_audio_s * self.sr_tagging))
        wav = torch.tensor(wav)

        return (
            wav.view(int(self.len_audio_s * self.sr_tagging)),
            {
                "name": os.path.basename(filename),
                "start": self.meta["start"][idx],
                "date": self.meta["date"][idx].strftime("%Y%m%d_%H%M%S"),
                "ecoac": ecoac,
            },
        )
        # return([None, None])

    def __len__(self):
        return len(self.meta["filename"])


def get_dataloader_site(
    meta_site, savepath, len_audio_s, save_audio=True, batch_size=12, num_workers=8
):

    meta_dataloader = pd.DataFrame(columns=["filename", "sr", "start", "stop"])

    total_files = len(meta_site["filename"])
    # Progress bar for file processing
    with tqdm(
        total=total_files,
        desc="Processing files",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as file_pbar:
        for idx, wavfile in enumerate(meta_site["filename"]):

            len_file = meta_site["length"][idx]
            sr_in = meta_site["sr"][idx]
            duration = len_file / sr_in
            nb_win = int(duration // len_audio_s)

            # Update file progress bar with current file info
            file_pbar.set_postfix(
                {
                    "file": os.path.basename(wavfile)[:30]
                    + ("..." if len(os.path.basename(wavfile)) > 30 else ""),
                    "segments": nb_win,
                    "duration": f"{duration:.1f}s",
                }
            )

            # Progress bar for segment creation within this file
            with tqdm(
                total=nb_win,
                desc=f"Segments for {os.path.basename(wavfile)[:20]}...",
                unit="segment",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]",
            ) as segment_pbar:
                # cut into segments
                for win in range(nb_win):
                    delta = datetime.timedelta(seconds=int((win * len_audio_s)))
                    curmeta = pd.DataFrame.from_dict(
                        {
                            "filename": [wavfile],
                            "sr": [sr_in],
                            "start": ([win * len_audio_s]),
                            "stop": [(win + 1) * len_audio_s],
                            "len": [len_file],
                            "date": meta_site["datetime"][idx] + delta,
                        }
                    )
                    meta_dataloader = pd.concat(
                        [meta_dataloader, curmeta], ignore_index=True
                    )
                    segment_pbar.update(1)

            file_pbar.update(1)

    total_segments = len(meta_dataloader)
    print(
        f"Segmentation complete: {total_segments} segments created from {total_files} files"
    )
    print(
        f"Total audio data: ~{total_segments * len_audio_s / 3600:.1f} hours of audio and Average segments per file: {total_segments/total_files:.1f}"
    )

    site_set = Silent_dataset(
        meta_dataloader.reset_index(drop=True), len_audio_s, savepath, save_audio
    )
    site_set = torch.utils.data.DataLoader(
        site_set, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    return site_set
