from pathlib import Path
from typing import Literal

import numpy as np
import librosa
import libfmp.c3
from pydantic import BaseModel, Field

from chord_rec.data_utils import Song


class ChromagramVersion(str):
    STFT = 'STFT'
    CQT = 'CQT'
    IIR = 'IIR'


class ChromagramConfig(BaseModel):
    version: Literal['STFT', 'CQT', 'IIR'] = 'STFT'
    sample_rate: int = Field(22050, gt=0)
    window_size: int = Field(4096, gt=0)
    hop_size: int = Field(2048, gt=0)
    gamma: float | None = Field(None, gt=0)
    norm: Literal['1', '2', 'max'] | None = '2'


def compute_chromagram(
    audio_path: Path | str,
    config: ChromagramConfig | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    """Compute a chromagram for a WAV file.

    Args:
        audio_path: Path to the audio file.
        config: Chromagram configuration. Defaults to STFT with standard parameters.

    Returns:
        X: Chromagram matrix (12 x N_frames).
        Fs_X: Feature rate of the chromagram (Hz).
        x: Raw audio signal.
        Fs: Sampling rate of the audio signal (Hz).
        x_dur: Duration of the audio signal in seconds.
    """
    if config is None:
        config = ChromagramConfig()

    x, Fs = librosa.load(str(audio_path), sr=config.sample_rate)
    x_dur = x.shape[0] / Fs
    N, H = config.window_size, config.hop_size

    if config.version == 'STFT':
        S = librosa.stft(x, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        S = np.log(1 + config.gamma * np.abs(S) ** 2) if config.gamma else np.abs(S) ** 2
        X = librosa.feature.chroma_stft(S=S, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    elif config.version == 'CQT':
        X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=H, norm=None)
    else:  # IIR
        S = librosa.iirt(y=x, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if config.gamma:
            S = np.log(1.0 + config.gamma * S)
        X = librosa.feature.chroma_cqt(
            C=S, bins_per_octave=12, n_octaves=7,
            fmin=librosa.midi_to_hz(24), norm=None,
        )

    if config.norm is not None:
        X = libfmp.c3.normalize_feature_sequence(X, norm=config.norm)

    Fs_X = Fs / H
    return X, Fs_X, x, Fs, x_dur


def compute_chromagram_batch(
    songs: list[Song],
    indices: list[int],
    config: ChromagramConfig | None = None,
    verbose: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
    """Compute chromagrams for a subset of songs.

    Args:
        songs: List of Song models.
        indices: Indices into songs to process.
        config: Chromagram configuration shared across all songs.
        verbose: Print progress if True.

    Returns:
        X_dict: Mapping from index to chromagram matrix.
        Fs_X_dict: Mapping from index to feature rate.
        x_dur_dict: Mapping from index to audio duration in seconds.
    """
    if config is None:
        config = ChromagramConfig()

    X_dict: dict[int, np.ndarray] = {}
    Fs_X_dict: dict[int, float] = {}
    x_dur_dict: dict[int, float] = {}

    for i in indices:
        song = songs[i]
        if verbose:
            print(f'Processing: {song.name} [{config.version}]')
        X, Fs_X, _, _, x_dur = compute_chromagram(song.audio_path, config)
        X_dict[i] = X
        Fs_X_dict[i] = Fs_X
        x_dur_dict[i] = x_dur

    return X_dict, Fs_X_dict, x_dur_dict
