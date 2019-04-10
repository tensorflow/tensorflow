# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Internal ops related to the Graphcore IPU."""
from functools import wraps
from tensorflow.python.platform import tf_logging as logging
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops


def remap(x, name=None):
  """Clone and map the input linearly across the IPU.

  Args:
    x: The tensor to the remap.
    name: Optional op name.

  Returns:
    A `Tensor` which is has been linearly mapped across the IPU.
  """

  logging.warning("remap is a Graphcore internal op")
  return gen_poputil_ops.ipu_remap(x, name=name)
