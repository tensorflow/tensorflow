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
"""Directives are special no-op functions that serve as compilation markers.

They provide static information like type hints, compilation and TensorFlow
overrides.

These serve as annotations in the compiled code, allowing the user some control
over the compilation process. They have no functional role at runtime.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export

UNSPECIFIED = object()


def set_element_type(entity, dtype, shape=UNSPECIFIED):
  """Indicates that the entity is expected hold items of specified type/shape.

  The staged TensorFlow ops will reflect and assert this data type. Ignored
  otherwise.

  Args:
    entity: The entity to annotate.
    dtype: TensorFlow dtype value to assert for entity.
    shape: Optional shape to assert for entity.
  """
  del entity
  del dtype
  del shape


@tf_export('autograph.experimental.set_loop_options')
def set_loop_options(
    parallel_iterations=UNSPECIFIED,
    back_prop=UNSPECIFIED,
    swap_memory=UNSPECIFIED,
    maximum_iterations=UNSPECIFIED):
  """Specifies additional arguments to be passed to the enclosing while_loop.

  The parameters apply to and only to the immediately enclosing loop. It only
  has effect if the loop is staged as a TF while_loop; otherwise the parameters
  have no effect.

  Usage example:

    @tf.function(autograph=True)
    def dynamic_rnn(..., parallel_iterations=32):
      num_steps = ...
      for t in tf.range(num_steps):
        tf.autograph.experimental.set_loop_options(
            parallel_iterations=parallel_iterations)
        ...

  Args:
    parallel_iterations: See tf.while_loop.
    back_prop: See tf.while_loop.
    swap_memory: See tf.while_loop.
    maximum_iterations: See tf.while_loop.
  """
  del parallel_iterations
  del back_prop
  del swap_memory
  del maximum_iterations
