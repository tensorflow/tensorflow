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

"""Operations for automatic batching and unbatching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_batch_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_batch_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util.tf_export import tf_export


@tf_export("nondifferentiable_batch_function")
def batch_function(num_batch_threads,
                   max_batch_size,
                   batch_timeout_micros,
                   allowed_batch_sizes=None,
                   max_enqueued_batches=10,
                   autograph=True,
                   enable_large_batch_splitting=True):
  """Batches the computation done by the decorated function.

  So, for example, in the following code

  ```python
  @batch_function(1, 2, 3)
  def layer(a):
    return tf.matmul(a, a)

  b = layer(w)
  ```

  if more than one session.run call is simultaneously trying to compute `b`
  the values of `w` will be gathered, non-deterministically concatenated
  along the first axis, and only one thread will run the computation. See the
  documentation of the `Batch` op for more details.

  Assumes that all arguments of the decorated function are Tensors which will
  be batched along their first dimension.

  SparseTensor is not supported. The return value of the decorated function
  must be a Tensor or a list/tuple of Tensors.

  Args:
    num_batch_threads: Number of scheduling threads for processing batches
     of work. Determines the number of batches processed in parallel.
    max_batch_size: Batch sizes will never be bigger than this.
    batch_timeout_micros: Maximum number of microseconds to wait before
     outputting an incomplete batch.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
     does nothing. Otherwise, supplies a list of batch sizes, causing the op
     to pad batches up to one of those sizes. The entries must increase
     monotonically, and the final entry must equal max_batch_size.
    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.
    autograph: Whether to use autograph to compile python and eager style code
     for efficient graph-mode execution.
    enable_large_batch_splitting: The value of this option doesn't affect
     processing output given the same input; it affects implementation details
     as stated below: 1. Improve batching efficiency by eliminating unnecessary
     adding. 2.`max_batch_size` specifies the limit of input and
     `allowed_batch_sizes` specifies the limit of a task to be processed. API
     user can give an input of size 128 when 'max_execution_batch_size'
     is 32 -> implementation can split input of 128 into 4 x 32, schedule
     concurrent processing, and then return concatenated results corresponding
     to 128.

  Returns:
    The decorated function will return the unbatched computation output Tensors.
  """

  def decorator(fn):  # pylint: disable=missing-docstring

    def decorated(*args):  # pylint: disable=missing-docstring

      @function.defun(autograph=autograph)
      def computation(*computation_args):
        return fn(*computation_args)

      computation = computation.get_concrete_function(
          *[tensor_spec.TensorSpec(dtype=x.dtype, shape=x.shape, name=str(i))
            for i, x in enumerate(args)])

      with ops.name_scope("batch") as name:
        for a in args:
          if not isinstance(a, ops.Tensor):
            raise ValueError("All arguments to functions decorated with "
                             "`batch_function`  are supposed to be Tensors; "
                             "found %s" % repr(a))
        return gen_batch_ops.batch_function(
            num_batch_threads=num_batch_threads,
            max_batch_size=max_batch_size,
            batch_timeout_micros=batch_timeout_micros,
            allowed_batch_sizes=allowed_batch_sizes,
            max_enqueued_batches=max_enqueued_batches,
            shared_name=name,
            enable_large_batch_splitting=enable_large_batch_splitting,
            f=computation,
            in_tensors=list(args),
            captured_tensors=computation.captured_inputs,
            Tout=[o.dtype for o in computation.outputs])

    return decorated

  return decorator
