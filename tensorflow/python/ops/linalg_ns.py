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

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops

# Linear algebra ops.
band_part = array_ops.matrix_band_part
cholesky = linalg_ops.cholesky
cholesky_solve = linalg_ops.cholesky_solve
det = linalg_ops.matrix_determinant
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
lstsq = linalg_ops.matrix_solve_ls
norm = linalg_ops.norm
qr = linalg_ops.qr
set_diag = array_ops.matrix_set_diag
solve = linalg_ops.matrix_solve
svd = linalg_ops.svd
tensordot = math_ops.tensordot
trace = math_ops.trace
transpose = array_ops.matrix_transpose
triangular_solve = linalg_ops.matrix_triangular_solve

# Additional aliases for consistency with root namespace for ops whose names
# changed significantly when moving to the linalg namespace.
self_adjoint_eig = linalg_ops.self_adjoint_eig
self_adjoint_eigvals = linalg_ops.self_adjoint_eigvals
solve_ls = linalg_ops.matrix_solve_ls

# Seal API.
del absolute_import
del array_ops
del division
del linalg_ops
del math_ops
del print_function
del special_math_ops
