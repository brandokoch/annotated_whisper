import torch 
from torch import Tensor
from typing import List, Tuple
import numpy as np 
import warnings
import tqdm

from .tokenizer import Tokenizer
from .architectures.transformer import AudioToTextTransformerWithKVCaching
from .utils import *
from .transforms import log_mel_spectrogram, pad_or_trim
from .configs import *
from .decoding import *


class Whisper:

    def __init__(self, model_name, configuration, tokenizer = None):

        # Fetch model 
        model_pth = fetch_whisper_model(model_name)

        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        configuration.device = self.device

        # Load model
        with (open(model_pth, "rb")) as fp:
            checkpoint = torch.load(fp, map_location=configuration.device)
        
        dims = ModelDimensions(**checkpoint["dims"])
        
        self.model = AudioToTextTransformerWithKVCaching(dims)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(configuration.device)

        # Set configuration
        if self.device == torch.device("cpu") and configuration.fp16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            configuration.fp16 = False
        self.dims = dims
        self.configuration = configuration
        self.is_multilingual = self.model.dims.n_vocab == 51865
        self.dtype = torch.float16 if self.configuration.fp16 else torch.float32
        self.sample_len = configuration.sample_len or self.model.dims.n_text_ctx // 2
        self.input_stride = self.configuration.audio_n_frames // self.model.dims.n_audio_ctx
        self.time_precision = self.input_stride * self.configuration.audio_hop_length / self.configuration.audio_sample_rate

        # Set Tokenizer
        if tokenizer is None:
            tokenizer = Tokenizer(
                self.is_multilingual,
                self.configuration.language,
                self.configuration.task,
            )
        self.tokenizer = tokenizer

        # Determine sot sequence
        if configuration.without_timestamps:
            self.sot_sequence = self.tokenizer.sot_sequence_including_notimestamps # e.g. '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>'
        else:
            self.sot_sequence: Tuple[int] = self.tokenizer.sot_sequence # e.g. '<|startoftranscript|><|en|><|transcribe|>'

        # Initialize decoder
        if self.configuration.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                self.configuration.beam_size, self.tokenizer.eot, self.model, self.configuration.patience
            )
        else:
            self.decoder = GreedyDecoder(self.configuration.temperature, self.tokenizer.eot)
        self.decoder.reset()

        #
        # Sequence ranker: implements how to rank a group of sampled sequences
        #
        self.sequence_ranker = MaximumLikelihoodRanker(self.configuration.length_penalty)


    def get_lang_toks_and_probs(self, mel: Tensor) -> Tuple[Tensor, List[dict]]:
        """
        Detect the spoken language in the audio, and return them as list of strings, along with the ids
        of the most probable language tokens and the probability distribution over all language tokens.
        This is performed outside the main decode loop in order to not interfere with kv-caching.

        Returns
        -------
        language_tokens : Tensor, shape = (n_audio,)
            ids of the most probable language tokens, which appears after the startoftranscript token.
        language_probs : List[Dict[str, float]], length = n_audio
            list of dictionaries containing the probability distribution over all languages.
        """

        if (self.tokenizer.language is None or self.tokenizer.language_token not in self.tokenizer.sot_sequence):
            raise ValueError("This model doesn't have language tokens so it can't perform lang id")

        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)

 
        mel = self.model.encode(mel) # (n_audio, n_audio_ctx, n_audio_state)

        # Forward pass using a single token: startoftranscript
        n_audio = mel.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(mel.device)  # (n_audio, 1)
        logits = self.model.decode(x, mel)[:, 0] # (n_audio, n_vocab)

        # Fetch logits for language tokens and get probability distribution over them
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)  # (n_vocab)
        mask[list(self.tokenizer.all_language_tokens)] = False 
        logits[:, mask] = -np.inf

        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()

        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        return language_tokens, language_probs
    
    def get_language_from_mel(self, mel: Tensor) -> str:

        # Using first 30 seconds of audio to detect language
        mel_segment = pad_or_trim(
            mel,
            self.configuration.audio_n_frames
        ).to(self.configuration.device).to(self.dtype) # (audio_n_mels, audio_n_frames)

        with torch.no_grad():
            _, probs = self.get_lang_toks_and_probs(mel_segment)
        language = max(probs, key=probs.get)

        return language
    
    def is_no_speech(self, no_speech_prob, avg_logprob):

        # "We found that the proba-
        # bility of the <|nospeech|> token alone is not sufficient
        # to distinguish a segment with no speech, but combining
        # the no-speech probability threshold of 0.6 and the average
        # log-probability threshold of −1 makes the voice activity
        # detection of Whisper more reliable.""
        if self.configuration.no_speech_threshold is not None:
            # no voice activity check
            speech_not_detected = no_speech_prob > self.configuration.no_speech_threshold
            if (
                self.configuration.logprob_threshold is not None
                and avg_logprob > self.configuration.logprob_threshold
            ):
                # don't skip if the logprob is high enough, despite the no_speech_prob
                speech_not_detected = False
        
        return speech_not_detected
    
    def extract_segments_and_modify_seek(self, tokens, seek, time_offset, segment_size, segment_duration, result):
        # shifting the window according to the
        # timestamps predicted by the model."

        segments = []

        timestamp_tokens_mask: torch.Tensor = tokens.ge(self.tokenizer.timestamp_begin)
        single_timestamp_ending = timestamp_tokens_mask[-2:].tolist() == [False, True]

        # Find pairs of timestamp tokens that are consecutive and get index of the second timestamp token
        consecutive = torch.where(timestamp_tokens_mask[:-1] & timestamp_tokens_mask[1:])[0]
        consecutive.add_(1)

        if len(consecutive) > 0:
            # If the output contains two consecutive timestamp tokens
            slices = consecutive.tolist()
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0].item() - self.tokenizer.timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - self.tokenizer.timestamp_begin
                segments.append(
                    self.new_segment(
                        seek = seek,
                        start=time_offset + start_timestamp_pos * self.time_precision,
                        end=time_offset + end_timestamp_pos * self.time_precision,
                        tokens=sliced_tokens,
                        result=result,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # Single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # Otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_pos = (
                    tokens[last_slice - 1].item() - self.tokenizer.timestamp_begin
                )
                seek += last_timestamp_pos * self.input_stride
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens_mask.nonzero().flatten()]
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != self.tokenizer.timestamp_begin
            ):
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = (
                    timestamps[-1].item() - self.tokenizer.timestamp_begin
                )
                duration = last_timestamp_pos * self.time_precision

            segments.append(
                self.new_segment(
                    seek = seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                    result=result,
                )
            )
            seek += segment_size
        
        return segments , seek


    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        initial_prompt = None
    ):


        #
        # Preprocessing audio to spectrogram
        #

        # Pad 30-seconds of silence to the input audio so we can for sure feed 30 sec into model
        mel = log_mel_spectrogram(
            audio,
            self.configuration.audio_n_mels,
            self.configuration.audio_n_fft,
            self.configuration.audio_hop_length,
            padding=self.configuration.audio_n_samples) # (audio_n_mels, -1)
        
        # Frames that contain speech without the part we padded
        content_frames = mel.shape[-1] - self.configuration.audio_n_frames  

        #
        # Detecting language 
        #

        if self.configuration.language is None:
            self.configuration.language = self.get_language_from_mel(mel)
            print(f"Detected language: {LANGUAGES[self.configuration.language].title()}")
            self.model.cleanup_caching()
        print("Language: ", self.configuration.language)

        self.tokenizer = Tokenizer(
            self.is_multilingual,
            self.configuration.language,
            self.configuration.task,
        )

        # Initial prompt can be used to improve transcription by providing context 
        all_tokens = []
        if initial_prompt is not None:
            initial_prompt_tokens = self.tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt_tokens)
        else:
            initial_prompt_tokens = []

        # "We developed a strategy to perform buffered transcrip-
        # tion of long audio by consecutively transcribing 30-second
        # segments of audio and shifting the window according to the
        # timestamps predicted by the model."
        audio_segment_seek = 0 
        all_segments = []
        prompt_reset_since = 0

        while audio_segment_seek < content_frames:
            mel_segment = mel[:, audio_segment_seek : audio_segment_seek + self.configuration.audio_n_frames] # (audio_n_mels, audio_n_frames)
            mel_segment = pad_or_trim(mel_segment, self.configuration.audio_n_frames).to(self.configuration.device).to(self.dtype) # ( audio_n_mels, audio_n_frames )

            # these are the previous prompts that are feeded after <|startofprev|> token and before <|startoftranscript|> token
            self.prompt = all_tokens[prompt_reset_since:]

            result: DecodingResult = self.decode_with_fallback(mel_segment)
            tokens = torch.tensor(result.tokens)

            # print(self.tokenizer.encoding.decode(tokens.tolist()))


            segment_size = min(self.configuration.audio_n_frames, content_frames - audio_segment_seek)
            speech_not_detected = self.is_no_speech(result.no_speech_prob, result.avg_logprob)
            if speech_not_detected:
                audio_segment_seek+= segment_size
                continue

            time_offset = float(audio_segment_seek * self.configuration.audio_hop_length / self.configuration.audio_sample_rate)
            segment_duration = segment_size * self.configuration.audio_hop_length / self.configuration.audio_sample_rate
            current_segments, audio_segment_seek = self.extract_segments_and_modify_seek(
                tokens, audio_segment_seek, time_offset, segment_size, segment_duration, result
            )

            # "Providing the tran-
            # scribed text from the preceding window as previous-text
            # conditioning when the applied temperature is below 0.5
            # further improves the performance.""
            if not self.configuration.condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # If a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

        
        return dict(
            text=self.tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
            segments=all_segments,
            language= self.configuration.language
        )
    
    def new_segment(
            self,
            seek,
            start: float,
            end: float,
            tokens: torch.Tensor,
            result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < self.tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": self.tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }
    
    def decode_with_fallback(self, segment: torch.Tensor) -> DecodingResult:

        # "We start with tem-
        # perature 0, i.e. always selecting the tokens with the high-
        # est probability, and increase the temperature by 0.2 up to
        # 1.0 when either the average log probability over the gen-
        # erated tokens is lower than −1 or the generated text has a
        # gzip compression rate higher than 2.4."

        decode_result = None
        for t in self.configuration.decode_with_fallback_temperatures:

            with torch.no_grad():

                if single := segment.ndim == 2:
                    new_seg = segment.unsqueeze(0) # 1, 80, 3000
                else:
                    new_seg = segment

                if t > 0:
                    result = self.decode(new_seg, t, disable_beam_size=True, disable_patience=True)
                else:
                    result = self.decode(new_seg, t, disable_best_of=True)

                decode_result = result[0] if single else result

            needs_fallback = False
            if (
                self.configuration.compression_ratio_threshold is not None
                and decode_result.compression_ratio > self.configuration.compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                self.configuration.logprob_threshold is not None
                and decode_result.avg_logprob < self.configuration.logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result



    @torch.no_grad()
    def decode(self, mel, temperature, disable_beam_size=False, disable_patience=False, disable_best_of=False) -> List[DecodingResult]:


        # Set decoding options
        beam_size = self.configuration.beam_size
        best_of = self.configuration.best_of
        patience = self.configuration.patience
        decoder = self.decoder 
        decoder.reset()

        # Handle case where decoding with fallback is being used
        if disable_beam_size or disable_best_of or disable_patience:
            beam_size = None if disable_beam_size else self.configuration.beam_size
            best_of = None if disable_best_of else self.configuration.best_of
            patience = None if disable_patience else self.configuration.patience

            if beam_size is not None:
                decoder = BeamSearchDecoder(
                    beam_size, self.tokenizer.eot, self.model, patience
                )
            else:
                decoder = GreedyDecoder(temperature, self.tokenizer.eot)
            decoder.reset()

        # Check if a valid decoding configuration is being used
        if beam_size is not None and best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if temperature == 0:
            if best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if patience is not None and beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if self.configuration.length_penalty is not None and not (
            0 <= self.configuration.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")


        #
        # Determine initial tokens
        #
        initial_tokens: Tuple[int] = self.tokenizer._get_initial_tokens(
            self.sot_sequence,
            self.sample_len,
            self.configuration.prefix,
            self.dims.n_text_ctx,
            self.prompt
        ) # e.g. '<|startoftranscript|><|en|><|transcribe|>'

        sot_index: int = initial_tokens.index(self.tokenizer.sot) 
        initial_token_length = len(initial_tokens) 

        #
        # Logit filters: applies various rules to suppress or penalize certain tokens
        #
        self.logit_filters= self.tokenizer._get_logit_filters(initial_token_length, self.model.dims.n_audio_ctx, self.configuration)

        n_audio = mel.shape[0]
        n_group = beam_size or best_of or 1

        if self.configuration.fp16:
            mel = mel.half() 

        #
        # Encoder
        #

        # Applies both audio encoder and transformer encoder
        audio_features = self.model.encode(mel) # (batch_size, 1500, 384), seems to be (batch_size, 1500, -1)

        if audio_features.dtype != (
            torch.float16 if self.configuration.fp16 else torch.float32
        ):
            return TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        #
        # Decoder
        # 
        tokens: Tensor = torch.tensor([initial_tokens]).repeat(n_audio, 1)

        languages = [self.configuration.language] * audio_features.shape[0]


        # call the main sampling loop
        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features_interleave = audio_features.repeat_interleave(n_group, dim=0) # (n_group, 1500, -1) 
        tok_seqs = tokens.repeat_interleave(n_group, dim=0).to(audio_features_interleave.device) 
        assert audio_features_interleave.shape[0] == tok_seqs.shape[0] # 5, 3
        n_batch = tok_seqs.shape[0]

        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features_interleave.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):

                # Use only the last token except in the first forward pass
                if tok_seqs.shape[-1] > initial_token_length: 
                    input_tokens = tok_seqs[:, -1:]  # (n_group, 1)
                else:
                    input_tokens = tok_seqs # (n_group, initial_token_length)

                # Predict
                logits = self.model.decode(input_tokens, audio_features_interleave) # (n_group, initial_token_length or 1, n_vocab)

                # Save no speech probs at the start of the transcription
                if (i == 0 and self.tokenizer.no_speech is not None): 
                    probs_at_sot = logits[:, sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # Extract predicted token
                last_token_logits = logits[:, -1] # (n_group, n_vocab)

                # Filter logits
                for logit_filter in self.logit_filters:
                    logit_filter.apply(last_token_logits, tok_seqs)

                # Decode
                tok_seqs, completed = decoder.update(tok_seqs, last_token_logits, sum_logprobs)

                # print(self.tokenizer.encoding.decode(tok_seqs[0].tolist()))

                if completed or tok_seqs.shape[-1] > self.dims.n_text_ctx:
                    break
        finally:
            self.model.cleanup_caching()

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: n_group]
        no_speech_probs = no_speech_probs[:: n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tok_seqs = tok_seqs.reshape(n_audio, n_group, -1)   # (n_audio, n_group, n_tokens)
        sum_logprobs = sum_logprobs.reshape(n_audio, n_group) # (n_audio, n_group)
        tok_seqs, sum_logprobs = decoder.finalize(tok_seqs, sum_logprobs) 

        tok_seqs: List[List[Tensor]] = [
            [group_seq[initial_token_length : (group_seq == self.tokenizer.eot).nonzero()[0, 0]] for group_seq in audio_seqs]
            for audio_seqs in tok_seqs
        ] # (n_audio, n_group, n_tokens)

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tok_seqs, sum_logprobs)

        tok_seqs: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tok_seqs)]
        texts: List[str] = [self.tokenizer.decode(t).strip() for t in tok_seqs]
        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tok_seqs, sum_logprobs)]

        fields = (
            texts,
            languages,
            tok_seqs,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tok_seqs,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tok_seqs, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


