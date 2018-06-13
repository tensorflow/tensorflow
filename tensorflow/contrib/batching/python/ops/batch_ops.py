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

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_batch_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_batch_ops import *
# pylint: enable=wildcard-import


@ops.RegisterGradient("Batch")
def _BatchGrad(op, *out_grads):  # pylint: disable=invalid-name
  """Gradient for batch op."""
  gradients = []
  for i in range(len(op.inputs)):
    gradients.append(
        gen_batch_ops.unbatch(
            out_grads[i],
            op.outputs[-2],
            op.outputs[-1],
            timeout_micros=op.get_attr("grad_timeout_micros"),
            shared_name="batch_gradient_{}_{}".format(op.name, i)))
  return gradients


@ops.RegisterGradient("Unbatch")
def _UnbatchGrad(op, grad):   # pylint: disable=invalid-name
  return [
      gen_batch_ops.unbatch_grad(
          op.inputs[0],
          op.inputs[1],
          grad,
          op.inputs[2],
          shared_name="unbatch_gradient_{}".format(op.name)), None, None
  ]


def batch_function(num_batch_threads,
                   max_batch_size,
                   batch_timeout_micros,
                   allowed_batch_sizes=None,
                   grad_timeout_micros=60 * 1000 * 1000,
                   unbatch_timeout_micros=60 * 1000 * 1000,
                   max_enqueued_batches=10):
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
    grad_timeout_micros: The timeout to use for the gradient. See the
     documentation of the unbatch op for more details. Defaults to 60s.
    unbatch_timeout_micros: The timeout to use for unbatching. See the
     documentation of the unbatch op for more details. Defaults to 60s.
    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.

  Returns:
    The decorated function will return the unbatched computation output Tensors.
  """

  def decorator(fn):  # pylint: disable=missing-docstring

    def decorated(*args):  # pylint: disable=missing-docstring
      types = [arg.dtype for arg in args]

      @function.Defun(*types)
      def computation(*computation_args):
        return fn(*computation_args)

      with ops.name_scope("batch") as name:
        for a in args:
          if not isinstance(a, ops.Tensor):
            raise ValueError("All arguments to functions decorated with "
                             "`batch_function`  are supposed to be Tensors; "
                             "found %s" % repr(a))
        for inp in computation.captured_inputs:
          print("inp: %s" % inp)
          for op in inp.consumers():
            print("op: %s" % op)
        return gen_batch_ops.batch_function(
            num_batch_threads=num_batch_threads,
            max_batch_size=max_batch_size,
            batch_timeout_micros=batch_timeout_micros,
            allowed_batch_sizes=allowed_batch_sizes,
            max_enqueued_batches=max_enqueued_batches,
            shared_name=name,
            f=computation,
            in_tensors=list(args),
            captured_tensors=computation.captured_inputs,
            Tout=[o.type for o in computation.definition.signature.output_arg])

    return decorated

  return decorator


def batch_function_v1(num_batch_threads,
                      max_batch_size,
                      batch_timeout_micros,
                      allowed_batch_sizes=None,
                      grad_timeout_micros=60 * 1000 * 1000,
                      unbatch_timeout_micros=60 * 1000 * 1000,
                      max_enqueued_batches=10):
  """Batches the computation done by the decorated function.

  This is the older version of batch_function(). Please use the former instead
  of this.

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
    grad_timeout_micros: The timeout to use for the gradient. See the
     documentation of the unbatch op for more details. Defaults to 60s.
    unbatch_timeout_micros: The timeout to use for unbatching. See the
     documentation of the unbatch op for more details. Defaults to 60s.
    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.

  Returns:
    The decorated function will return the unbatched computation output Tensors.
  """
  def decorator(f):  # pylint: disable=missing-docstring
    def decorated(*args):
      with ops.name_scope("batch") as name:
        for a in args:
          if not isinstance(a, ops.Tensor):
            raise ValueError("All arguments to functions decorated with "
                             "`batch_function`  are supposed to be Tensors; "
                             "found %s" % repr(a))
        batched_tensors, batch_index, id_t = gen_batch_ops.batch(
            args,
            num_batch_threads=num_batch_threads,
            max_batch_size=max_batch_size,
            batch_timeout_micros=batch_timeout_micros,
            max_enqueued_batches=max_enqueued_batches,
            allowed_batch_sizes=allowed_batch_sizes,
            grad_timeout_micros=grad_timeout_micros,
            shared_name=name)
        outputs = f(*batched_tensors)
        if isinstance(outputs, ops.Tensor):
          outputs_list = [outputs]
        else:
          outputs_list = outputs
        with ops.name_scope("unbatch") as unbatch_name:
          unbatched = [
              gen_batch_ops.unbatch(t, batch_index, id_t,
                                    timeout_micros=unbatch_timeout_micros,
                                    shared_name=unbatch_name + "/" + t.name)
              for t in outputs_list]
        if isinstance(outputs, ops.Tensor):
          return unbatched[0]
        return unbatched
    return decorated
  return decorator
