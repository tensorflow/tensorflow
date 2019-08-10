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
"""Public API for tf.linalg namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import,unused-import
from tensorflow.python.ops.linalg import adjoint_registrations as _adjoint_registrations
from tensorflow.python.ops.linalg import cholesky_registrations as _cholesky_registrations
from tensorflow.python.ops.linalg import inverse_registrations as _inverse_registrations
from tensorflow.python.ops.linalg import linear_operator_algebra as _linear_operator_algebra
from tensorflow.python.ops.linalg import matmul_registrations as _matmul_registrations
from tensorflow.python.ops.linalg import solve_registrations as _solve_registrations
from tensorflow.python.ops.linalg.linalg_impl import *
from tensorflow.python.ops.linalg.linear_operator import *
from tensorflow.python.ops.linalg.linear_operator_block_diag import *
from tensorflow.python.ops.linalg.linear_operator_circulant import *
from tensorflow.python.ops.linalg.linear_operator_composition import *
from tensorflow.python.ops.linalg.linear_operator_diag import *
from tensorflow.python.ops.linalg.linear_operator_full_matrix import *
from tensorflow.python.ops.linalg.linear_operator_identity import *
from tensorflow.python.ops.linalg.linear_operator_kronecker import *
from tensorflow.python.ops.linalg.linear_operator_low_rank_update import *
from tensorflow.python.ops.linalg.linear_operator_lower_triangular import *
from tensorflow.python.ops.linalg.linear_operator_toeplitz import *
from tensorflow.python.ops.linalg.linear_operator_zeros import *
# pylint: enable=wildcard-import

# Seal API.
# pylint: disable=undefined-variable
del absolute_import
del division
del print_function
del ops
del array_ops
del gen_linalg_ops
del linalg_ops
del math_ops
del special_math_ops
del tf_export
# pylint: enable=undefined-variable
