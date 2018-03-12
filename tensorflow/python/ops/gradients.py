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
"""Implements the graph generation for computation of gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.gradients_impl import AggregationMethod
from tensorflow.python.ops.gradients_impl import gradients
from tensorflow.python.ops.gradients_impl import hessians
# pylint: enable=unused-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    # TODO(drpng): find a good place to reference this.
    "AggregationMethod",
    "custom_gradient",
    "gradients",  # tf.gradients.gradients.
    "hessians",  # tf.gradients.hessians
]
remove_undocumented(__name__, _allowed_symbols)
