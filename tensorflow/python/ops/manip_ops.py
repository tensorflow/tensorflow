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
"""Operators for manipulating tensors.

API docstring: tensorflow.manip
"""

from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=protected-access
@tf_export('roll', v1=['roll', 'manip.roll'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('manip.roll')
def roll(input, shift, axis, name=None):  # pylint: disable=redefined-builtin
  return _gen_manip_ops.roll(input, shift, axis, name)


roll.__doc__ = _gen_manip_ops.roll.__doc__
# pylint: enable=protected-access
