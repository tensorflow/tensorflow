# Copyright 2015 Google Inc. All Rights Reserved.
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
# =============================================================================

"""Functional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_functional_ops import *
# pylint: enable=wildcard-import
# pylint: disable=unused-import
from tensorflow.python.ops.gen_functional_ops import _symbolic_gradient
# pylint: enable=unused-import


@ops.RegisterShape("SymbolicGradient")
def _symbolic_gradient_shape(op):
  # Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
  # (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
  # outputs (dx, dy, dz) are the same as (x, y, z).
  return [op.inputs[i].get_shape() for i in range(len(op.outputs))]
