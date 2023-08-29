import os
import base64
from typing import List, Tuple
from functools import cached_property

import tiktoken
from tiktoken_ext.openai_public import gpt2

from .utils import LANGUAGES, TO_LANGUAGE_CODE
from .decoding import *


class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    def __init__(self, multilingual: bool, language = None, task = None):

        # Deduce settings which are compatible with the model
        if language is not None:
            language = language.lower()
            if language not in LANGUAGES:
                if language in TO_LANGUAGE_CODE:
                    language = TO_LANGUAGE_CODE[language]
                else:
                    raise ValueError(f"Unsupported language: {language}")

        if multilingual:
            self.language = language or "en"
            self.task = task or "transcribe"
        else: 
            self.language = None
            self.task = None


        # Load encoding 
        self.encoding, self.special_tokens = self.get_encoding(multilingual)

        # Get sot sequence
        langs = list(LANGUAGES.keys())
        sot_sequence = [self.sot]
        sot_sequence.append(self.special_tokens[f"<|{self.language}|>"])
        if task is not None:
            task_token: int = self.transcribe if self.task == "transcribe" else self.translate
            sot_sequence.append(task_token)

        self.sot_sequence =  tuple(sot_sequence)

        print("Using sot_sequence: ", self.sot_sequence)
        print("Using decoded sot_sequence: ", self.decode(self.sot_sequence))


    def get_encoding(self, multilingual):

        if multilingual:
            encoding_name = "multilingual"
        else:
            encoding_name = "gpt2"

        # Initialize the encoding 
        vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{encoding_name}.tiktoken")
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in open(vocab_path) if line)
        }
        n_vocab = len(ranks)
        special_tokens = {}



        # "For timestamp predic-
        # tion, we predict time relative to the current audio segment,
        # quantizing all times to the nearest 20 milliseconds which
        # matches the native time resolution of Whisper models, and
        # add additional tokens to our vocabulary for each of these.""
        specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            *[f"<|{i * 0.02:.2f}|>" for i in range(1501)], # timesteps 0 to 30 
        ]

        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1

        encoding = tiktoken.Encoding(
            name=os.path.basename(vocab_path),
            explicit_n_vocab=n_vocab,
            pat_str=gpt2()["pat_str"],
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )

        return encoding, special_tokens


    def _get_suppress_tokens(self, additional_suppress_tokens) -> Tuple[int]:
        
        assert isinstance(additional_suppress_tokens, list), "suppress_tokens must be a list"

        additional_suppress_tokens.extend(self.non_speech_tokens)

        additional_suppress_tokens.extend(
            [
                self.transcribe,
                self.translate,
                self.sot,
                self.sot_prev,
                self.sot_lm,
            ]
        )

        if self.no_speech is not None:
            additional_suppress_tokens.append(self.no_speech)

        return tuple(sorted(set(additional_suppress_tokens)))

    def _get_initial_tokens(self, sot_sequence, sample_len, prefix, n_text_ctx, prompt) -> Tuple[int]:
        tokens = list(sot_sequence)

        if prefix := prefix:
            prefix_tokens = (self.encode(" " + prefix.strip())  if isinstance(prefix, str) else prefix) 
            
            if sample_len is not None:
                max_prefix_len = n_text_ctx // 2 - sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := prompt:
            prompt_tokens = (
                self.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.sot_prev]
                + prompt_tokens[-(n_text_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens) # e.g. ids of '<|startofprev|>Some spoken text.<|startoftranscript|><|en|><|transcribe|>'


    def _get_logit_filters(self, initial_token_length, n_audio_ctx, configuration):
        logit_filters = []
        if configuration.suppress_blank:
            logit_filters.append(SuppressBlank(self, initial_token_length)) 
        if configuration.suppress_common_tokens:
            logit_filters.append(SuppressTokens(self._get_suppress_tokens(configuration.suppress_additional_tokens)))
        if not configuration.without_timestamps:
            precision = configuration.audio_chunk_length / n_audio_ctx  # 0.02 seconds
            max_initial_timestamp_index = None
            if configuration.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    configuration.max_initial_timestamp / precision
                )
            logit_filters.append(
                ApplyTimestampRules(
                    self, initial_token_length, max_initial_timestamp_index 
                )
            )
        return logit_filters

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        if token := self.special_tokens.get(f"<|{self.language}|>", None):
            return token

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]: 
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}

        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

