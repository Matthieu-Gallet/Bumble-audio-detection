import os
import argparse
from tqdm import tqdm
import pandas as pd
from utils import metadata
from utils import dataloader
from utils.tagging_validation import tagging_validate
from models import PANNS_Model, inference
import json
import signal
from datetime import datetime
import sys


def save_checkpoint(save_path, processed_files, current_batch_idx, total_batches):
    """Save processing state to checkpoint file."""
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": list(processed_files),
        "current_batch_idx": current_batch_idx,
        "total_batches": total_batches,
        "completion_percentage": (
            (current_batch_idx / total_batches * 100) if total_batches > 0 else 0
        ),
    }

    checkpoint_path = os.path.join(save_path, "processing_checkpoint.json")
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def load_checkpoint(save_path):
    """Load processing state from checkpoint file."""
    checkpoint_path = os.path.join(save_path, "processing_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {"processed_files": [], "current_batch_idx": 0, "total_batches": 0}


def print_progress_bar(current, total, prefix="Progress", suffix="", length=50):
    """Print a progress bar to the console."""
    percent = (current / total * 100) if total > 0 else 0
    filled_length = int(length * current // total) if total > 0 else 0
    bar = "█" * filled_length + "░" * (length - filled_length)

    sys.stdout.write(f"\r{prefix}: |{bar}| {percent:.1f}% {suffix}")
    sys.stdout.flush()

    if current == total:
        print()  # New line when complete


def get_processed_files_from_checkpoint(checkpoint_data):
    """Extract processed files from checkpoint data."""
    return set(checkpoint_data.get("processed_files", []))


def is_file_processed(file_path, processed_files):
    """Check if a file has already been processed based on its path."""
    return file_path in processed_files


def main():
    parser = argparse.ArgumentParser(
        description="Script to process sound files recorded by Audiomoth "
    )
    parser.add_argument(
        "--data_path", default="example/audio/0002/", type=str, help="Path to wav files"
    )
    parser.add_argument(
        "--save_path",
        default="example/metadata/",
        type=str,
        help="Path to save meta data",
    )
    parser.add_argument("--name", default="", type=str, help="name of measurement")
    parser.add_argument("--audio_format", default="wav", type=str, help="wav or flac")
    parser.add_argument(
        "--l",
        default=10,
        type=int,
        help="Window length in seconds for audio tagging / must be more than 5 seconds",
    )
    parser.add_argument(
        "--model_type",
        default="MobileNetV2",
        type=str,
        help="Type of the model (e.g., ResNet22, MobileNetV2)",
    )
    parser.add_argument(
        "--save_audio_flac",
        default=1,
        type=int,
        help="Saving audio in flac format (needed to run visualization tool)",
    )
    parser.add_argument(
        "--multiprocessing",
        default=1,
        type=int,
        help="Number of processes to use for data loading",
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, help="Size of the batch for data loading"
    )
    args = parser.parse_args()

    AUDIO_FORMAT = args.audio_format
    LEN_AUDIO = args.l

    if LEN_AUDIO < 5:
        raise ValueError("With tagging, length_audio_segment must be more than 5")

    csvfile = os.path.join(args.save_path, f"indices_{args.name}.csv")
    audio_savepath = os.path.join(args.save_path, f"audio_{args.name}")

    # Create directories
    if not os.path.exists(audio_savepath):
        os.makedirs(audio_savepath)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(args.save_path)
    processed_files = get_processed_files_from_checkpoint(checkpoint_data)
    current_batch_idx = checkpoint_data.get("current_batch_idx", 0)

    if processed_files:
        print(
            f"Resuming from checkpoint: {len(processed_files)} files already processed"
        )

    # Initialize total_batches (will be set properly after dataloader creation)
    total_batches = 0
    signal_handler_called = False

    def signal_handler(signum, frame):
        nonlocal signal_handler_called
        if signal_handler_called:
            return  # Prevent multiple calls
        signal_handler_called = True

        # Ignore further signals immediately
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # print(f"\n⚠️  Received interrupt signal {signum}. Saving checkpoint...")
        save_checkpoint(
            args.save_path, processed_files, current_batch_idx, total_batches
        )
        # print("Checkpoint saved. Exiting...")

        # Force immediate exit without cleanup
        os._exit(1)

    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # get meta data file
    df_files = metadata.metadata_generator(args.data_path, AUDIO_FORMAT)
    if len(df_files) == 0:
        # raise('No audio file found')
        raise Exception("No audio file found")

    # get data loader
    dl = dataloader.get_dataloader_site(
        df_files,
        savepath=audio_savepath,
        len_audio_s=LEN_AUDIO,
        save_audio=args.save_audio_flac,
        batch_size=args.batch_size,
        num_workers=args.multiprocessing,
    )

    # Calculate total batches for progress tracking
    total_batches = len(dl)

    # Initialize data structures
    df_site = {"datetime": [], "name": [], "flacfile": [], "start": []}
    df_site["clipwise_output"] = []
    df_site["embedding"] = []
    df_site["sorted_indexes"] = []
    df_site["dB"] = []

    # Store ecoac keys for later use
    ecoac_keys = None

    ## Initialize audio tagging model
    model = PANNS_Model.from_pretrained(f"nicofarr/panns_{args.model_type}")
    model.eval()

    # Resume from checkpoint if available
    if processed_files:
        print(f"Skipping {len(processed_files)} already processed files")

    # Process batches with enhanced progress tracking
    processed_count = 0
    batch_count = 0

    print(
        f"Starting batch processing with {args.multiprocessing} workers for {total_batches} total batches to process"
    )

    for batch_idx, (inputs, info) in enumerate(
        tqdm(dl, desc="Processing batches", unit="batch")
    ):
        batch_count = batch_idx + 1
        current_batch_idx = batch_count

        # Save checkpoint at the beginning of each batch (in case of interruption)
        if batch_count == 1 or batch_count % 10 == 0:
            save_checkpoint(args.save_path, processed_files, batch_count, total_batches)

        # Check if this batch contains already processed files
        batch_files = []
        for idx, date_ in enumerate(info["date"]):
            file_key = f"{info['name'][idx]}_{date_}_{info['start'][idx]}"
            batch_files.append(file_key)

        # Skip batch if all files are already processed
        if all(file_key in processed_files for file_key in batch_files):
            continue

        # Show batch processing details
        batch_size = len(info["date"])
        # tqdm.write(f"📦 Processing batch {batch_count}/{total_batches} ({batch_size} segments)")

        # Process the batch
        clipwise_output, labels, sorted_indexes, embedding = inference(
            model, inputs, usecuda=False
        )

        # Store ecoac keys from first batch
        if ecoac_keys is None:
            ecoac_keys = list(info["ecoac"].keys())

        # Process each file in the batch
        for idx, date_ in enumerate(info["date"]):
            file_key = f"{info['name'][idx]}_{date_}_{info['start'][idx]}"

            # Skip if already processed
            if file_key in processed_files:
                continue

            df_site["datetime"].append(str(date_))
            df_site["name"].append(str(info["name"][idx]))
            df_site["flacfile"].append(str(date_) + ".flac")
            df_site["start"].append(float(info["start"][idx]))

            df_site["clipwise_output"].append(clipwise_output[idx])
            df_site["sorted_indexes"].append(sorted_indexes[idx])
            df_site["embedding"].append(embedding[idx])

            for key in ecoac_keys:
                df_site[key].append(float(info["ecoac"][key].numpy()[idx]))

            # Mark file as processed
            processed_files.add(file_key)
            processed_count += 1

        # Save checkpoint more frequently (after each batch to avoid data loss)
        save_checkpoint(args.save_path, processed_files, batch_count, total_batches)

        # Update progress
        print_progress_bar(
            batch_count,
            total_batches,
            prefix="Batch Progress",
            suffix=f"({processed_count} files processed)",
        )

    print(
        f"\n✅ Processing complete! Processed {processed_count} files in {batch_count} batches"
    )

    # Clean up checkpoint file on successful completion
    checkpoint_path = os.path.join(args.save_path, "processing_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file cleaned up")

    Df_tagging = tagging_validate(df_site)

    ## Dataframe with only ecoacoustic indices and important metadata

    Df_eco = pd.DataFrame()
    Df_eco["name"] = df_site["name"]
    Df_eco["start"] = df_site["start"]
    Df_eco["datetime"] = df_site["datetime"]
    if ecoac_keys:
        for key in ecoac_keys:
            Df_eco[key] = df_site[key]

    ## Fusing with the dataframe containing only the ecoacoustic indices
    Df_final = pd.merge(Df_tagging, Df_eco, on=["name", "start", "datetime"])

    Df_final.sort_values(by=["datetime", "start"]).to_csv(csvfile, index=False)
    print(f"Saved indices to {csvfile}")


if __name__ == "__main__":
    main()
