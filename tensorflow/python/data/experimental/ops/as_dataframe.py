# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied over from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/as_dataframe.py - all code belongs to original authors

import tensorflow as tf
import numpy as np
import dataclasses

from tensorflow_datasets.core import dataset_info
from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import features
from tensorflow_datasets.core import lazy_imports_lib
from tensorflow_datasets.core.utils import py_utils
from tensorflow_datasets.core.utils import type_utils
