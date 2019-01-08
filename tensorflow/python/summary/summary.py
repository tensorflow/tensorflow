# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tensor summaries for exporting information about a model.

See the [Summary](https://tensorflow.org/api_guides/python/summary) guide.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import json_format as _json_format

# exports Summary, SummaryDescription, Event, TaggedRunMetadata, SessionLog
# pylint: disable=unused-import
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.framework.summary_pb2 import SummaryDescription
from tensorflow.core.framework.summary_pb2 import SummaryMetadata as _SummaryMetadata  # pylint: enable=unused-import
from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.core.util.event_pb2 import TaggedRunMetadata
# pylint: enable=unused-import

from tensorflow.python.eager import context as _context
from tensorflow.python.framework import constant_op as _constant_op
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import gen_logging_ops as _gen_logging_ops
from tensorflow.python.ops import gen_summary_ops as _gen_summary_ops  # pylint: disable=unused-import
from tensorflow.python.ops import summary_op_util as _summary_op_util

# exports FileWriter, FileWriterCache
# pylint: disable=unused-import
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache
# pylint: enable=unused-import

from tensorflow.python.util import compat as _compat
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['summary.scalar'])
def scalar(name, tensor, collections=None, family=None):
  """Outputs a `Summary` protocol buffer containing a single scalar value.

  The generated Summary has a Tensor.proto containing the input Tensor.

  Args:
    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A real numeric Tensor containing a single value.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

  Raises:
    ValueError: If tensor has the wrong shape or type.
  """
  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  with _summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = _gen_logging_ops.scalar_summary(tags=tag, values=tensor, name=scope)
    _summary_op_util.collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


@tf_export(v1=['summary.image'])
def image(name, tensor, max_outputs=3, collections=None, family=None):
  """Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_outputs` summary values containing images. The
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

  The `tag` in the outputted Summary.Value protobufs is generated based on the
  name, with a suffix depending on the max_outputs setting:

  *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*name*/image/0', '*name*/image/1', etc.

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
      width, channels]` where `channels` is 1, 3, or 4.
    max_outputs: Max number of batch elements to generate images for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  with _summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = _gen_logging_ops.image_summary(
        tag=tag, tensor=tensor, max_images=max_outputs, name=scope)
    _summary_op_util.collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


@tf_export(v1=['summary.histogram'])
def histogram(name, values, collections=None, family=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with a histogram.

  Adding a histogram summary makes it possible to visualize your data's
  distribution in TensorBoard. You can see a detailed explanation of the
  TensorBoard histogram dashboard
  [here](https://www.tensorflow.org/get_started/tensorboard_histograms).

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    values: A real numeric `Tensor`. Any shape. Values to use to
      build the histogram.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  with _summary_op_util.summary_scope(
      name, family, values=[values],
      default_name='HistogramSummary') as (tag, scope):
    val = _gen_logging_ops.histogram_summary(
        tag=tag, values=values, name=scope)
    _summary_op_util.collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


@tf_export(v1=['summary.audio'])
def audio(name, tensor, sample_rate, max_outputs=3, collections=None,
          family=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
  `sample_rate`.

  The `tag` in the outputted Summary.Value protobufs is generated based on the
  name, with a suffix depending on the max_outputs setting:

  *  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*name*/audio/0', '*name*/audio/1', etc

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
      signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  with _summary_op_util.summary_scope(
      name, family=family, values=[tensor]) as (tag, scope):
    sample_rate = _ops.convert_to_tensor(
        sample_rate, dtype=_dtypes.float32, name='sample_rate')
    val = _gen_logging_ops.audio_summary_v2(
        tag=tag, tensor=tensor, max_outputs=max_outputs,
        sample_rate=sample_rate, name=scope)
    _summary_op_util.collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


@tf_export(v1=['summary.text'])
def text(name, tensor, collections=None):
  """Summarizes textual data.

  Text data summarized via this plugin will be visible in the Text Dashboard
  in TensorBoard. The standard TensorBoard Text Dashboard will render markdown
  in the strings, and will automatically organize 1d and 2d tensors into tables.
  If a tensor with more than 2 dimensions is provided, a 2d subarray will be
  displayed along with a warning message. (Note that this behavior is not
  intrinsic to the text summary api, but rather to the default TensorBoard text
  plugin.)

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: a string-type Tensor to summarize.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

  Returns:
    A TensorSummary op that is configured so that TensorBoard will recognize
    that it contains textual data. The TensorSummary is a scalar `Tensor` of
    type `string` which contains `Summary` protobufs.

  Raises:
    ValueError: If tensor has the wrong type.
  """
  if tensor.dtype != _dtypes.string:
    raise ValueError('Expected tensor %s to have dtype string, got %s' %
                     (tensor.name, tensor.dtype))

  summary_metadata = _SummaryMetadata(
      plugin_data=_SummaryMetadata.PluginData(plugin_name='text'))
  t_summary = tensor_summary(
      name=name,
      tensor=tensor,
      summary_metadata=summary_metadata,
      collections=collections)
  return t_summary


@tf_export(v1=['summary.tensor_summary'])
def tensor_summary(name,
                   tensor,
                   summary_description=None,
                   collections=None,
                   summary_metadata=None,
                   family=None,
                   display_name=None):
  """Outputs a `Summary` protocol buffer with a serialized tensor.proto.

  Args:
    name: A name for the generated node. If display_name is not set, it will
      also serve as the tag name in TensorBoard. (In that case, the tag
      name will inherit tf name scopes.)
    tensor: A tensor of any type and shape to serialize.
    summary_description: A long description of the summary sequence. Markdown
      is supported.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    summary_metadata: Optional SummaryMetadata proto (which describes which
      plugins may use the summary value).
    family: Optional; if provided, used as the prefix of the summary tag,
      which controls the name used for display on TensorBoard when
      display_name is not set.
    display_name: A string used to name this data in TensorBoard. If this is
      not set, then the node name will be used instead.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """

  if summary_metadata is None:
    summary_metadata = _SummaryMetadata()

  if summary_description is not None:
    summary_metadata.summary_description = summary_description

  if display_name is not None:
    summary_metadata.display_name = display_name

  serialized_summary_metadata = summary_metadata.SerializeToString()

  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  with _summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = _gen_logging_ops.tensor_summary_v2(
        tensor=tensor,
        tag=tag,
        name=scope,
        serialized_summary_metadata=serialized_summary_metadata)
    _summary_op_util.collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


@tf_export(v1=['summary.merge'])
def merge(inputs, collections=None, name=None):
  # pylint: disable=line-too-long
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
      added to these collections. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer resulting from the merging.

  Raises:
    RuntimeError: If called with eager mode enabled.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  """
  # pylint: enable=line-too-long
  if _context.executing_eagerly():
    raise RuntimeError(
        'Merging tf.summary.* ops is not compatible with eager execution. '
        'Use tf.contrib.summary instead.')
  if _summary_op_util.skip_summary():
    return _constant_op.constant('')
  name = _summary_op_util.clean_tag(name)
  with _ops.name_scope(name, 'Merge', inputs):
    val = _gen_logging_ops.merge_summary(inputs=inputs, name=name)
    _summary_op_util.collect(val, collections, [])
  return val


@tf_export(v1=['summary.merge_all'])
def merge_all(key=_ops.GraphKeys.SUMMARIES, scope=None, name=None):
  """Merges all summaries collected in the default graph.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.
    scope: Optional scope used to filter the summary ops, using `re.match`

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  """
  if _context.executing_eagerly():
    raise RuntimeError(
        'Merging tf.summary.* ops is not compatible with eager execution. '
        'Use tf.contrib.summary instead.')
  summary_ops = _ops.get_collection(key, scope=scope)
  if not summary_ops:
    return None
  else:
    return merge(summary_ops, name=name)


@tf_export(v1=['summary.get_summary_description'])
def get_summary_description(node_def):
  """Given a TensorSummary node_def, retrieve its SummaryDescription.

  When a Summary op is instantiated, a SummaryDescription of associated
  metadata is stored in its NodeDef. This method retrieves the description.

  Args:
    node_def: the node_def_pb2.NodeDef of a TensorSummary op

  Returns:
    a summary_pb2.SummaryDescription

  Raises:
    ValueError: if the node is not a summary op.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  """

  if node_def.op != 'TensorSummary':
    raise ValueError("Can't get_summary_description on %s" % node_def.op)
  description_str = _compat.as_str_any(node_def.attr['description'].s)
  summary_description = SummaryDescription()
  _json_format.Parse(description_str, summary_description)
  return summary_description
