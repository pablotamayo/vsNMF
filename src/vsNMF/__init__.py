from os.path import abspath
import datetime
import numpy as np
x = datetime.datetime.now()

VERSION = '1.0'
print('vsNMF library version {} @ {} date: {}'.format(VERSION, abspath(__file__), x))

from .vsNMF import vsNMF
from .normalize_dataset_vs_reference import normalize_dataset_vs_reference
from .vsNMF import compute_row_sparseness
