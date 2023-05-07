import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:


    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
    """
    Produce all the start and end index positions of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which
    the models will be trained.
    Returns a tuple with four elements:
    1) The index position of the first element to be included in the input sequence
    2) The index position of the last element to be included in the input sequence
    3) The index position of the first element to be included in the target sequence
    4) The index position of the last element to be included in the target sequence

    Args:
        num_obs (int): Number of observations in the entire dataset for which
                        indices must be generated.
        input_len (int): Length of the input sequence (a sub-sequence of
                         of the entire data sequence)
        step_size (int): Size of each step as the data sequence is traversed.
                         If 1, the first sub-sequence will be indices 0-input_len,
                         and the next will be 1-input_len.
        forecast_horizon (int): How many index positions is the target away from
                                the last index position of the input sequence?
                                If forecast_horizon=1, and the input sequence
                                is data[0:10], the target will be data[11:taget_len].
        target_len (int): Length of the target / output sequence.
    """

    input_len = round(input_len)  # just a precaution
    start_position = 0
    stop_position = num_obs - 1  # because of 0 indexing

    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    target_first_idx = subseq_last_idx + forecast_horizon
    target_last_idx = target_first_idx + target_len
    print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while target_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len

    return indices
