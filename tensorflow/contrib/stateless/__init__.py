# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Stateless random ops which take seed as a tensor input.

Instead of taking `seed` as an attr which initializes a mutable state within
the op, these random ops take `seed` as an input, and the random numbers are
a deterministic function of `shape` and `seed`.

WARNING: These ops are in contrib, and are not stable.  They should be
consistent across multiple runs on the same hardware, but only for the same
version of the code.

@@stateless_multinomial
@@stateless_random_uniform
@@stateless_random_normal
@@stateless_truncated_normal
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_stateless_random_ops import *

from tensorflow.python.util.all_util import remove_undocumented

ops.NotDifferentiable("StatelessMultinomial")
ops.NotDifferentiable("StatelessRandomNormal")
ops.NotDifferentiable("StatelessRandomUniform")
ops.NotDifferentiable("StatelessTruncatedNormal")

remove_undocumented(__name__)
