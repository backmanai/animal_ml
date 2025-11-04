"""
Data loading helper functions for audio classification project.

This module provides functions for loading and validating audio data.
"""

import librosa
from pathlib import Path


def load_animal_audio_stats(animal_name, data_dir='../data'):
    """
    Load audio statistics for a given animal category.

    Args:
        animal_name: Name of the animal (e.g., 'cats', 'dogs', 'birds')
        data_dir: Path to the data directory (default: '../data')

    Returns:
        dict: Contains durations, sample_rates, failed_files, and problematic_files

    Raises:
        FileNotFoundError: If data directory doesn't exist
        NotADirectoryError: If path exists but is not a directory
    """
    durations = []
    sample_rates = []
    failed_files = []
    problematic_files = []

    # Check if the data directory exists
    data_path = Path(data_dir) / animal_name
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path.absolute()}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {data_path.absolute()}")

    # Get all wav files
    wav_files = list(data_path.glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_path.absolute()}")

    print(f"Found {len(wav_files)} {animal_name} sound .wav files in {data_path}")

    # Process each file
    print(f"Getting sample rate and duration for each {animal_name} file...")
    for wav_file in wav_files:
        try:
            audio, sr = librosa.load(wav_file, sr=None)  # Load with original sample-rate

            # Check if audio data is valid
            if len(audio) == 0:
                problematic_files.append(str(wav_file))
                print(f"{wav_file.name} loaded but contains no audio data")
                continue

            duration = len(audio) / sr

            # Check for suspiciously short files (less than 0.1 seconds)
            if duration < 0.1:
                problematic_files.append(str(wav_file))
                print(f"{wav_file.name} has suspicious duration: {duration:.3f}s")

            durations.append(duration)
            sample_rates.append(sr)

        except Exception as e:
            failed_files.append(str(wav_file))
            print(f"Error loading {wav_file.name}: {e}")

    return {
        'durations': durations,
        'sample_rates': sample_rates,
        'failed_files': failed_files,
        'problematic_files': problematic_files,
        'total_files': len(wav_files),
        'successful_files': len(durations)
    }
