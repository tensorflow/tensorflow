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
# =============================================================================

"""Helper context for running models with bfloat16."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


def _get_custom_getter():
  """Returns a custom getter that this class's methods must be called under.

  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:

  ```python
  network = ConvNetBuilder(...)
  with tf.compat.v1.variable_scope('cg',
                                   custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```

  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

  def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_bfloat16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == dtypes.bfloat16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs['dtype'] = dtypes.float32
      cast_to_bfloat16 = True
    var = getter(*args, **kwargs)
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    if cast_to_bfloat16:
      var = math_ops.cast(var, dtypes.bfloat16)
    return var

  return inner_custom_getter


@tf_export(v1=['tpu.bfloat16_scope'])
@tf_contextlib.contextmanager
def bfloat16_scope(name=None):
  """Scope class for bfloat16 variables so that the model uses custom getter.

  This enables variables to be read as bfloat16 type when using get_variable.
  """
  if name is None:
    name = ''
  with variable_scope.variable_scope(
      name, custom_getter=_get_custom_getter()) as varscope:
    yield varscope
