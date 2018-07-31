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
# =============================================================================
"""cond_v2 wrapper module.

This imports the cond_v2 method and all necessary dependencies (this is to avoid
circular dependencies in the cond_v2 implementation). See cond_v2_impl for more
information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.framework import function
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.ops import gradients_impl

from tensorflow.python.ops.cond_v2_impl import cond_v2
# pylint: enable=unused-import
