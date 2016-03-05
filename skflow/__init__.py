"""Main Scikit Flow module."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pkg_resources as pkg_rs
import numpy as np
import tensorflow as tf

from skflow.io import *
from skflow.estimators import *
from skflow import ops
from skflow import preprocessing
from skflow.io import data_feeder
from skflow import models
from skflow.trainer import TensorFlowTrainer


__version__ = pkg_rs.get_distribution("skflow").version

sklearn_version = pkg_rs.get_distribution("scikit-learn").version
tensorflow_version = pkg_rs.get_distribution("tensorflow").version

if float(sklearn_version) < 0.16:
    raise ImportError("Your scikit-learn version needs to be at least 0.16. ")
if float(tensorflow_version[0:3]) < 0.7:
    raise ImportError("Your tensorflow version needs to be at least 0.7. ")
