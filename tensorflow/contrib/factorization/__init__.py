# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Ops and modules related to factorization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.factorization.python.ops.clustering_ops import *
from tensorflow.contrib.factorization.python.ops.factorization_ops import *
from tensorflow.contrib.factorization.python.ops.gmm import *
from tensorflow.contrib.factorization.python.ops.gmm_ops import *
from tensorflow.contrib.factorization.python.ops.wals import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'KMeans',
    'COSINE_DISTANCE',
    'KMEANS_PLUS_PLUS_INIT',
    'RANDOM_INIT',
    'SQUARED_EUCLIDEAN_DISTANCE',
    'WALSModel',
    'GMM',
    'gmm',
    'GmmAlgorithm',
    'WALSMatrixFactorization',
]

remove_undocumented(__name__, _allowed_symbols)
