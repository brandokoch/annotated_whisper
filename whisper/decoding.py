from typing import List, Optional, Tuple, Sequence

import torch
import numpy as np

from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

class MaximumLikelihoodRanker:
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def scores(self, logprobs, lengths):
        result = []
        for logprob, length in zip(logprobs, lengths):
            if self.length_penalty is None:
                penalty = length
            else:
                # from the Google NMT paper
                penalty = ((5 + length) / 6) ** self.length_penalty
            result.append(logprob / penalty)
        return result

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):

        # tokens shape is (n_audio, n_group, n_tokens)
        # sum_logprobs shape is (n_audio, n_group)

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        result =  [np.argmax(self.scores(p, l)) for p, l in zip(sum_logprobs, lengths)]
        return result


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        model,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.model = model
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.final_tokens = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.final_tokens = None # list of dictonaries that map a tok seq to score

    def update(
        self, tok_seqs: Tensor, tok_preds_logit: Tensor, tok_seq_logprob_sum: Tensor
    ) -> Tuple[Tensor, bool]:
    
        
        # works also for multiple audio inputs with each having beam_size candidates
        if tok_seqs.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tok_seqs.shape}[0] % {self.beam_size} != 0")

        audio_cnt = tok_seqs.shape[0] // self.beam_size
        if self.final_tokens is None:  # for the first update
            self.final_tokens = [{} for _ in range(audio_cnt)]

        # taking the log softmax of logits (we can do summation instead of multiplication now)  
        tok_preds_logprob = F.log_softmax(tok_preds_logit.float(), dim=-1)  

        new_tok_seqs, source_indices, list_of_finished_tok_seq_to_score = [], [], []
        for audio_idx in range(audio_cnt):

            #  seq to logprob, seq to beam_idx, seq to logprob
            new_tok_seq_to_logprob_sum= {}
            new_tok_seq_to_beam_idx= {} 
            finished_tok_seq_to_score = {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            # for each beam get new candidates and put them to tokens_to_score
            for beam_idx in range(self.beam_size):

                audio_beam_idx = audio_idx * self.beam_size + beam_idx
                existing_beam_tokens = tok_seqs[audio_beam_idx].tolist()

                # for each beam generate new beam_size candidates and store
                tok_pred_logprob_and_tok_pairs = zip(*tok_preds_logprob[audio_beam_idx].topk(self.beam_size + 1))
                for tok_pred_logprob, tok in tok_pred_logprob_and_tok_pairs:
                    new_tok_seq_logprob = (tok_seq_logprob_sum[audio_beam_idx] + tok_pred_logprob).item()
                    new_tok_seq = tuple(existing_beam_tokens + [tok.item()])

                    new_tok_seq_to_logprob_sum[new_tok_seq] = new_tok_seq_logprob
                    new_tok_seq_to_beam_idx[new_tok_seq] = audio_beam_idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            # rank by score and keep the top beam_size sequences for each audio, also store those that finished
            saved = 0
            for new_tok_seq in sorted(new_tok_seq_to_logprob_sum, key=new_tok_seq_to_logprob_sum.get, reverse=True):
                if new_tok_seq[-1] == self.eot:
                    # Storing finished
                    finished_tok_seq_to_score[new_tok_seq] = new_tok_seq_to_logprob_sum[new_tok_seq]
                else:
                    # Preparing beam_size unfinished ones to next iteration
                    tok_seq_logprob_sum[len(new_tok_seqs)] = new_tok_seq_to_logprob_sum[new_tok_seq]
                    new_tok_seqs.append(new_tok_seq)
                    source_indices.append(new_tok_seq_to_beam_idx[new_tok_seq])

                    saved += 1
                    if saved == self.beam_size:
                        break

            list_of_finished_tok_seq_to_score.append(finished_tok_seq_to_score)

        tok_seqs = torch.tensor(new_tok_seqs, device=tok_seqs.device)
        self.model.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.final_tokens) == len(list_of_finished_tok_seq_to_score)
        for previously_finished, newly_finished in zip(self.final_tokens, list_of_finished_tok_seq_to_score):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.final_tokens
        )
        return tok_seqs, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.final_tokens):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.final_tokens
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.final_tokens
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin: int): 
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        # tokens.shape is (n_batch, current_sequence_length)
        # logits.shape is (n_batch, vocab_size)

        # Suppreses blank token and end of transcript token at the beggining of transcription
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                logits[k, self.tokenizer.timestamp_begin : timestamps[-1]] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf



