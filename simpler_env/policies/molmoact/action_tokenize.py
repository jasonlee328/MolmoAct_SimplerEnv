import numpy as np
from transformers import Qwen2Tokenizer
import numpy as np

class ActionTokenizer:
    def __init__(self, tokenizer: Qwen2Tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps them to tokens at the end of vocab.
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Converts token IDs back to continuous actions using the bin centers.
        """
        action_token_ids = np.array([
            self.tokenizer.vocab_size if token is None else token for token in action_token_ids
        ])
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]