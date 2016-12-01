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

"""Helpers to manipulate a tensor graph in python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=unused-import
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
from tensorflow.python.framework.graph_util_impl import extract_sub_graph
from tensorflow.python.framework.graph_util_impl import must_run_on_cpu
from tensorflow.python.framework.graph_util_impl import remove_training_nodes
from tensorflow.python.framework.graph_util_impl import set_cpu0
from tensorflow.python.framework.graph_util_impl import tensor_shape_from_node_def_name
# pylint: enable=unused-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    # TODO(drpng): find a good place to reference this.
    "convert_variables_to_constants",
    "extract_sub_graph",
    "must_run_on_cpu",
    "set_cpu0",
    "tensor_shape_from_node_def_name",
    "remove_training_nodes",
]
remove_undocumented(__name__, _allowed_symbols)
