from transformers.modeling_outputs import ModelOutput
from transformers.configuration_bart import BartConfig
from typing import List, Optional, Tuple
import bisect
from dataclasses import dataclass

import torch

@dataclass
class EncDecQaMimBartOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    qa_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    summary_loss: Optional[torch.FloatTensor] = None
    mim_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BartSummaryOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    nll_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None