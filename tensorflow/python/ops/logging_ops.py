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

"""Logging and Summary Operations."""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_logging_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
def Assert(condition, data, summarize=None, name=None):
  """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  NOTE: To ensure that Assert executes, one usually attaches a dependency:

  ```python
   # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  x = tf.with_dependencies([assert_op], x)
  ```

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).

  Returns:
    assert_op: An `Operation` that, when executed, raises a
    `tf.errors.InvalidArgumentError` if `condition` is not true.
  """
  return gen_logging_ops._assert(condition, data, summarize, name)


ops.RegisterShape("Assert")(common_shapes.no_outputs)


def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Args:
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    message: A string, prefix of the error message.
    first_n: Only log `first_n` number of times. Negative numbers log always;
             this is the default.
    summarize: Only print this many entries of each tensor. If None, then a
               maximum of 3 elements are printed per input tensor.
    name: A name for the operation (optional).

  Returns:
    Same tensor as `input_`.
  """
  return gen_logging_ops._print(input_, data, message, first_n, summarize, name)


@ops.RegisterGradient("Print")
def _PrintGrad(op, *grad):
  return list(grad) + [None] * (len(op.inputs) - 1)


ops.RegisterShape("Print")(common_shapes.unchanged_shape)


def _Collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


def histogram_summary(tag, values, collections=None, name=None):
  """Outputs a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    tag: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
    values: A real numeric `Tensor`. Any shape. Values to use to
      build the histogram.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  with ops.op_scope([tag, values], name, "HistogramSummary") as scope:
    val = gen_logging_ops._histogram_summary(
        tag=tag, values=values, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


def image_summary(tag, tensor, max_images=3, collections=None, name=None):
  """Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  Args:
    tag: A scalar `Tensor` of type `string`. Used to build the `tag`
      of the summary values.
    tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
      width, channels]` where `channels` is 1, 3, or 4.
    max_images: Max number of batch elements to generate images for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  with ops.op_scope([tag, tensor], name, "ImageSummary") as scope:
    val = gen_logging_ops._image_summary(
        tag=tag, tensor=tensor, max_images=max_images, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


def audio_summary(tag,
                  tensor,
                  sample_rate,
                  max_outputs=3,
                  collections=None,
                  name=None):
  """Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
  `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    tag: A scalar `Tensor` of type `string`. Used to build the `tag`
      of the summary values.
    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
    sample_rate: The sample rate of the signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  with ops.op_scope([tag, tensor], name, "AudioSummary") as scope:
    val = gen_logging_ops._audio_summary(tag=tag,
                                         tensor=tensor,
                                         max_outputs=max_outputs,
                                         sample_rate=sample_rate,
                                         name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


def merge_summary(inputs, collections=None, name=None):
  """Merges summaries.

  This op creates a
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of `string` `Tensor` objects containing serialized `Summary`
      protocol buffers.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer resulting from the merging.
  """
  with ops.op_scope(inputs, name, "MergeSummary") as scope:
    val = gen_logging_ops._merge_summary(inputs=inputs, name=name)
    _Collect(val, collections, [])
  return val


def merge_all_summaries(key=ops.GraphKeys.SUMMARIES):
  """Merges all summaries collected in the default graph.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
  summary_ops = ops.get_collection(key)
  if not summary_ops:
    return None
  else:
    return merge_summary(summary_ops)


def get_summary_op():
  """Returns a single Summary op that would run all summaries.

  Either existing one from `SUMMARY_OP` collection or merges all existing
  summaries.

  Returns:
    If no summaries were collected, returns None. Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
  summary_op = ops.get_collection(ops.GraphKeys.SUMMARY_OP)
  if summary_op is not None:
    if summary_op:
      summary_op = summary_op[0]
    else:
      summary_op = None
  if summary_op is None:
    summary_op = merge_all_summaries()
    if summary_op is not None:
      ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
  return summary_op


def scalar_summary(tags, values, collections=None, name=None):
  """Outputs a `Summary` protocol buffer with scalar values.

  The input `tags` and `values` must have the same shape.  The generated
  summary has a summary value for each tag-value pair in `tags` and `values`.

  Args:
    tags: A `string` `Tensor`.  Tags for the summaries.
    values: A real numeric Tensor.  Values for the summaries.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  with ops.op_scope([tags, values], name, "ScalarSummary") as scope:
    val = gen_logging_ops._scalar_summary(tags=tags, values=values, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


ops.NoGradient("HistogramAccumulatorSummary")
ops.NoGradient("HistogramSummary")
ops.NoGradient("ImageSummary")
ops.NoGradient("AudioSummary")
ops.NoGradient("MergeSummary")
ops.NoGradient("ScalarSummary")


@ops.RegisterShape("HistogramAccumulatorSummary")
@ops.RegisterShape("HistogramSummary")
@ops.RegisterShape("ImageSummary")
@ops.RegisterShape("AudioSummary")
@ops.RegisterShape("MergeSummary")
@ops.RegisterShape("ScalarSummary")
def _ScalarShape(unused_op):
  return [tensor_shape.scalar()]
