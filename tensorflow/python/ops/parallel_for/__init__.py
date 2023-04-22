# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for pfor, for_loop, jacobian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.parallel_for import *  # pylint: disable=wildcard-import
from tensorflow.python.ops.parallel_for.control_flow_ops import for_loop
from tensorflow.python.ops.parallel_for.control_flow_ops import pfor
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
from tensorflow.python.ops.parallel_for.gradients import jacobian
