# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Asserts and Boolean Checks for RaggedTensors."""

from tensorflow.python.ops import check_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(check_ops.assert_type)
def assert_type(tensor: ragged_tensor.Ragged, tf_type, message=None, name=None):
  return check_ops.assert_type(tensor.flat_values, tf_type,
                               message=message, name=name)


