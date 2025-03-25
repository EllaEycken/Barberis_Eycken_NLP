"""Additional functions used in this repository."""
# Code based on Helena Balabin

## Imports
import re
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

from ella_phd_nlp_project.ella_phd_nlp_code.constants import Q_FILE_MAPPINGS

##