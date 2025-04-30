import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError


from fermbot.common.constants import HF_FERMBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats 

