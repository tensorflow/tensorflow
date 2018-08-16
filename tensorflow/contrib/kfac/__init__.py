# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Kronecker-factored Approximate Curvature Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long
from tensorflow.contrib.kfac.python.ops import curvature_matrix_vector_products_lib as curvature_matrix_vector_products
from tensorflow.contrib.kfac.python.ops import estimator_lib as estimator
from tensorflow.contrib.kfac.python.ops import fisher_blocks_lib as fisher_blocks
from tensorflow.contrib.kfac.python.ops import fisher_factors_lib as fisher_factors
from tensorflow.contrib.kfac.python.ops import layer_collection_lib as layer_collection
from tensorflow.contrib.kfac.python.ops import loss_functions_lib as loss_functions
from tensorflow.contrib.kfac.python.ops import op_queue_lib as op_queue
from tensorflow.contrib.kfac.python.ops import optimizer_lib as optimizer
from tensorflow.contrib.kfac.python.ops import utils_lib as utils
from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long

_allowed_symbols = [
    "curvature_matrix_vector_products",
    "estimator",
    "fisher_blocks",
    "fisher_factors",
    "layer_collection",
    "loss_functions",
    "op_queue",
    "optimizer",
    "utils",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
