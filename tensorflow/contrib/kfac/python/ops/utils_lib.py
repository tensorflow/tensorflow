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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.contrib.kfac.python.ops.utils import *
from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long,wildcard-import

_allowed_symbols = [
    "SequenceDict",
    "tensors_to_column",
    "column_to_tensors",
    "kronecker_product",
    "layer_params_to_mat2d",
    "mat2d_to_layer_params",
    "compute_pi",
    "posdef_inv",
    "posdef_inv_matrix_inverse",
    "posdef_inv_cholesky",
    "posdef_inv_funcs",
    "SubGraph",
    "generate_random_signs",
    "fwd_gradients",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
