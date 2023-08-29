from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from torch import Tensor
    

@dataclass
class Configuration:

    # General audio
    audio_sample_rate=16000
    audio_n_fft=400 # window length
    audio_n_mels=80
    audio_hop_length=160
    audio_chunk_length=30 # 30 seconds 
    audio_n_samples=480000  # CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk (everything is expected to be 30 sec for input)
    audio_n_frames=3000  # 3000 frames in a mel spectrogram input

    # Model
    model_type: str = "tiny"

    # Tokenizer
    prompt: Optional[Union[str, List[int]]] = None
    prefix: Optional[Union[str, List[int]]] = None
    suppress_common_tokens: bool = True
    suppress_additional_tokens: Optional[List[int]] = field(default_factory=list)
    suppress_blank: bool = True

    # General
    task: str = "transcribe" # transcribe, translate, lang_id
    language: Optional[str] = None
    compression_ratio_threshold : Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6

    # Decoding
    temperature: Optional[float] = 0.0
    decode_with_fallback_temperatures: Tuple[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    best_of: Optional[int] = 5 # only applicable if temp is zero
    beam_size: Optional[int] = 5 # only applicable if temp is not zero
    patience: Optional[float] = None # only applicable if temp is not zero
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0
    
    sample_len: Optional[int] = None
    length_penalty: Optional[float] = None

    fp16: bool = True

    # Other
    condition_on_previous_text: bool = True

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int