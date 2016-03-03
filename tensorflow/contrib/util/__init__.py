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
# ==============================================================================

"""Utilities for dealing with Tensors.

## Miscellaneous Utility Functions

@@constant_value
@@make_tensor_proto
@@make_ndarray
@@stripped_op_list_for_graph

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.framework.tensor_util import constant_value
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray
from tensorflow.python.training.saver import stripped_op_list_for_graph
from tensorflow.python.util.all_util import make_all


__all__ = make_all(__name__)
