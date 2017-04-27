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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_logging_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util.deprecation import deprecated

# The python wrapper for Assert is in control_flow_ops, as the Assert
# call relies on certain conditionals for its dependencies.  Use
# control_flow_ops.Assert.


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Note: This op prints to the standard error. It is not currently compatible
    with jupyter notebook (printing to the notebook *server's* output, not into
    the notebook).

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


def _Collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


@deprecated(
    "2016-11-30", "Please switch to tf.summary.histogram. Note that "
    "tf.summary.histogram uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in.")
def histogram_summary(tag, values, collections=None, name=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with a histogram.

  This ops is deprecated. Please switch to tf.summary.histogram.

  For an explanation of why this op was deprecated, and information on how to
  migrate, look ['here'](https://www.tensorflow.org/code/tensorflow/contrib/deprecated/__init__.py)

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
  with ops.name_scope(name, "HistogramSummary", [tag, values]) as scope:
    val = gen_logging_ops._histogram_summary(
        tag=tag, values=values, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


@deprecated(
    "2016-11-30", "Please switch to tf.summary.image. Note that "
    "tf.summary.image uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in. Also, the max_images "
    "argument was renamed to max_outputs.")
def image_summary(tag, tensor, max_images=3, collections=None, name=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with images.

  For an explanation of why this op was deprecated, and information on how to
  migrate, look ['here'](https://www.tensorflow.org/code/tensorflow/contrib/deprecated/__init__.py)

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
  with ops.name_scope(name, "ImageSummary", [tag, tensor]) as scope:
    val = gen_logging_ops._image_summary(
        tag=tag, tensor=tensor, max_images=max_images, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


@deprecated(
    "2016-11-30", "Please switch to tf.summary.audio. Note that "
    "tf.summary.audio uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in.")
def audio_summary(tag,
                  tensor,
                  sample_rate,
                  max_outputs=3,
                  collections=None,
                  name=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with audio.

  This op is deprecated. Please switch to tf.summary.audio.
  For an explanation of why this op was deprecated, and information on how to
  migrate, look ['here'](https://www.tensorflow.org/code/tensorflow/contrib/deprecated/__init__.py)

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
    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
      signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  with ops.name_scope(name, "AudioSummary", [tag, tensor]) as scope:
    sample_rate = ops.convert_to_tensor(sample_rate, dtype=dtypes.float32,
                                        name="sample_rate")
    val = gen_logging_ops._audio_summary_v2(tag=tag,
                                            tensor=tensor,
                                            max_outputs=max_outputs,
                                            sample_rate=sample_rate,
                                            name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


@deprecated("2016-11-30", "Please switch to tf.summary.merge.")
def merge_summary(inputs, collections=None, name=None):
  # pylint: disable=line-too-long
  """Merges summaries.

  This op is deprecated. Please switch to tf.summary.merge, which has identical
  behavior.

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
  with ops.name_scope(name, "MergeSummary", inputs):
    val = gen_logging_ops._merge_summary(inputs=inputs, name=name)
    _Collect(val, collections, [])
  return val


@deprecated("2016-11-30", "Please switch to tf.summary.merge_all.")
def merge_all_summaries(key=ops.GraphKeys.SUMMARIES):
  """Merges all summaries collected in the default graph.

  This op is deprecated. Please switch to tf.summary.merge_all, which has
  identical behavior.

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


@deprecated(
    "2016-11-30", "Please switch to tf.summary.scalar. Note that "
    "tf.summary.scalar uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in. Also, passing a "
    "tensor or list of tags to a scalar summary op is no longer "
    "supported.")
def scalar_summary(tags, values, collections=None, name=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with scalar values.

  This ops is deprecated. Please switch to tf.summary.scalar.
  For an explanation of why this op was deprecated, and information on how to
  migrate, look ['here'](https://www.tensorflow.org/code/tensorflow/contrib/deprecated/__init__.py)

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
  with ops.name_scope(name, "ScalarSummary", [tags, values]) as scope:
    val = gen_logging_ops._scalar_summary(tags=tags, values=values, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


ops.NotDifferentiable("HistogramSummary")
ops.NotDifferentiable("ImageSummary")
ops.NotDifferentiable("AudioSummary")
ops.NotDifferentiable("AudioSummaryV2")
ops.NotDifferentiable("MergeSummary")
ops.NotDifferentiable("ScalarSummary")
