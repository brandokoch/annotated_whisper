
import os
from functools import lru_cache
from typing import Optional, Union

import torch 
import numpy as np
from torch.nn import functional as F


#
# Audio processing
#


def pad_or_trim(array, length: int, *, axis: int = -1): 
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:  
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    n_mels: int, 
    n_fft: int, 
    hop_length: int, 
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio) # (n_samples)

    if device is not None:
        audio = audio.to(device)

    if padding > 0:
        audio = F.pad(audio, (0, padding)) # (n_samples + padding)

    window = torch.hann_window(n_fft).to(audio.device)  # (n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True) # (n_fft // 2 + 1, n_frames)
    magnitudes = stft[..., :-1].abs() ** 2      # (n_fft //2 + 1, n_frames)

    filters = mel_filters(audio.device, n_mels)  # (audio_n_mels, n_fft // 2 + 1)
    mel_spec = filters @ magnitudes  # (audio_n_mels, n_frames)

    log_spec = torch.clamp(mel_spec, min=1e-10).log10() # (audio_n_mels, n_frames)
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0) # (audio_n_mels, n_frames)
    log_spec = (log_spec + 4.0) / 4.0 # (audio_n_mels, n_frames)
    return log_spec 