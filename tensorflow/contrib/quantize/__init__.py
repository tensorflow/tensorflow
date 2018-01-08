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
"""Functions for rewriting graphs for quantized training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long
from tensorflow.contrib.quantize.python.quantize_graph import *
# pylint: enable=unused-import,wildcard-import,line-too-long

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "create_eval_graph",
    "create_training_graph",
]

remove_undocumented(__name__, _allowed_symbols)
