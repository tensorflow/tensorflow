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
"""Graph editor module allows to modify an existing graph in place.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.graph_editor import edit
from tensorflow.contrib.graph_editor import match
from tensorflow.contrib.graph_editor import reroute
from tensorflow.contrib.graph_editor import select
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.contrib.graph_editor import transform
from tensorflow.contrib.graph_editor import util

from tensorflow.contrib.graph_editor.edit import bypass
from tensorflow.contrib.graph_editor.edit import connect

# edit: detach
from tensorflow.contrib.graph_editor.edit import detach
from tensorflow.contrib.graph_editor.edit import detach_inputs
from tensorflow.contrib.graph_editor.edit import detach_outputs

# edit: reroute
from tensorflow.contrib.graph_editor.reroute import reroute_a2b
from tensorflow.contrib.graph_editor.reroute import reroute_a2b_inputs
from tensorflow.contrib.graph_editor.reroute import reroute_a2b_outputs
from tensorflow.contrib.graph_editor.reroute import reroute_b2a
from tensorflow.contrib.graph_editor.reroute import reroute_b2a_inputs
from tensorflow.contrib.graph_editor.reroute import reroute_b2a_outputs
from tensorflow.contrib.graph_editor.reroute import swap
from tensorflow.contrib.graph_editor.reroute import swap_inputs
from tensorflow.contrib.graph_editor.reroute import swap_outputs

from tensorflow.contrib.graph_editor.select import select_ops
from tensorflow.contrib.graph_editor.select import select_ts

from tensorflow.contrib.graph_editor.subgraph import SubGraphView

from tensorflow.contrib.graph_editor.transform import copy
from tensorflow.contrib.graph_editor.transform import Transformer


# some useful aliases
ph = util.make_placeholder_from_dtype_and_shape
sgv = subgraph.make_view
sgv_scope = subgraph.make_view_from_scope
ts = select.select_ts
ops = select.select_ops
matcher = match.OpMatcher

