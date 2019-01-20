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

import pprint
import random
import sys

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

# The python wrapper for Assert is in control_flow_ops, as the Assert
# call relies on certain conditionals for its dependencies.  Use
# control_flow_ops.Assert.


# Assert and Print are special symbols in python, so we must
# have an upper-case version of them.
#
# For users with Python 3 or Python 2.7
# with `from __future__ import print_function`, we could also allow lowercase.
# See https://github.com/tensorflow/tensorflow/issues/18053


# pylint: disable=invalid-name
@deprecated("2018-08-20", "Use tf.print instead of tf.Print. Note that "
                          "tf.print returns a no-output operator that directly "
                          "prints the output. Outside of defuns or eager mode, "
                          "this operator will not be executed unless it is "
                          "directly specified in session.run or used as a "
                          "control dependency for other operators. This is "
                          "only a concern in graph mode. Below is an example "
                          "of how to ensure tf.print executes in graph mode:\n"
                          """```python
    sess = tf.Session()
    with sess.as_default():
        tensor = tf.range(10)
        print_op = tf.print(tensor)
        with tf.control_dependencies([print_op]):
          out = tf.add(tensor, tensor)
        sess.run(out)
    ```
Additionally, to use tf.print in python 2.7, users must make sure to import
the following:

  `from __future__ import print_function`
""")
@tf_export(v1=["Print"])
def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op (behaves like `tf.identity`) with the side effect
  of printing `data` when evaluating.

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
    A `Tensor`. Has the same type and contents as `input_`.
  """
  return gen_logging_ops._print(input_, data, message, first_n, summarize, name)
# pylint: enable=invalid-name


def _generate_placeholder_string(x, default_placeholder="{}"):
  """Generate and return a string that does not appear in `x`."""
  placeholder = default_placeholder
  rng = random.Random(5)
  while placeholder in x:
    placeholder = placeholder + str(rng.randint(0, 9))
  return placeholder


def _is_filepath(output_stream):
  """Returns True if output_stream is a file path."""
  return isinstance(output_stream, str) and output_stream.startswith("file://")


# Temporarily disable pylint g-doc-args error to allow giving more context
# about what the kwargs are.
# Because we are using arbitrary-length positional arguments, python 2
# does not support explicitly specifying the keyword arguments in the
# function definition.
# pylint: disable=g-doc-args
@tf_export("print")
def print_v2(*inputs, **kwargs):
  """Print the specified inputs.

  Returns an operator that prints the specified inputs to a desired
  output stream or logging level. The inputs may be dense or sparse Tensors,
  primitive python objects, data structures that contain Tensors, and printable
  python objects. Printed tensors will recursively show the first and last
  `summarize` elements of each dimension.

  With eager execution enabled and/or inside a `tf.contrib.eager.defun` this
  operator will automatically execute, and users only need to call `tf.print`
  without using the return value. When constructing graphs outside of a
  `tf.contrib.eager.defun`, one must either include the returned op
  in the input to `session.run`, or use the operator as a control dependency for
  executed ops by specifying `with tf.control_dependencies([print_op])`.

  @compatibility(python2)
  In python 2.7, make sure to import the following:
  `from __future__ import print_function`
  @end_compatibility

  Example:
    Single-input usage:
    ```python
    tf.enable_eager_execution()
    tensor = tf.range(10)
    tf.print(tensor, output_stream=sys.stderr)
    ```
    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)

    Multi-input usage:
    ```python
    tf.enable_eager_execution()
    tensor = tf.range(10)
    tf.print("tensors:", tensor, {2: tensor * 2}, output_stream=sys.stdout)
    ```
    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to
    sys.stdout)

    Usage in a defun:
    ```python
    tf.enable_eager_execution()

    @tf.contrib.eager.defun
    def f():
        tensor = tf.range(10)
        tf.print(tensor, output_stream=sys.stderr)
        return tensor

    range_tensor = f()
    ```
    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)

    Usage when constructing graphs:
    ```python
    sess = tf.Session()
    with sess.as_default():
        tensor = tf.range(10)
        print_op = tf.print("tensors:", tensor, {2: tensor * 2},
                            output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
          tripled_tensor = tensor * 3
        sess.run(tripled_tensor)
    ```
    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to
    sys.stdout)

  Note: This op is only partially compatible with Jupyter notebooks and colabs.
    Because it prints to the C++ standard out / standard error, this will go
    in the notebook kernel's console output, not in the notebook cell output.

  Args:
    *inputs: Positional arguments that are the inputs to print. Inputs in the
      printed output will be separated by spaces. Inputs may be python
      primitives, tensors, data structures such as dicts and lists that
      may contain tensors (with the data structures possibly nested in
      arbitrary ways), and printable python objects.
    output_stream: The output stream, logging level, or file to print to.
      Defaults to sys.stderr, but sys.stdout, tf.logging.info,
      tf.logging.warning, and tf.logging.error are also supported. To print to
      a file, pass a string started with "file://" followed by the file path,
      e.g., "file:///tmp/foo.out".
    summarize: The first and last `summarize` elements within each dimension are
      recursively printed per Tensor. If None, then the first 3 and last 3
      elements of each dimension are printed for each tensor. If set to -1, it
      will print all elements of every tensor.
    name: A name for the operation (optional).

  Returns:
    A print operator that prints the specified inputs in the specified output
    stream or logging level.

  Raises:
    ValueError: If an unsupported output stream is specified.
  """
  # Because we are using arbitrary-length positional arguments, python 2
  # does not support explicitly specifying the keyword arguments in the
  # function definition. So, we manually get the keyword arguments w/ default
  # values here.
  output_stream = kwargs.pop("output_stream", sys.stderr)
  name = kwargs.pop("name", None)
  summarize = kwargs.pop("summarize", 3)
  if kwargs:
    raise ValueError("Unrecognized keyword arguments for tf.print: %s" % kwargs)
  format_name = None
  if name:
    format_name = name + "_format"

  # Match the C++ string constants representing the different output streams.
  # Keep this updated!
  output_stream_to_constant = {
      sys.stdout: "stdout",
      sys.stderr: "stderr",
      tf_logging.INFO: "log(info)",
      tf_logging.info: "log(info)",
      tf_logging.WARN: "log(warning)",
      tf_logging.warning: "log(warning)",
      tf_logging.warn: "log(warning)",
      tf_logging.ERROR: "log(error)",
      tf_logging.error: "log(error)",
  }

  if _is_filepath(output_stream):
    output_stream_string = output_stream
  else:
    output_stream_string = output_stream_to_constant.get(output_stream)
    if not output_stream_string:
      raise ValueError(
          "Unsupported output stream, logging level, or file." +
          str(output_stream) + ". Supported streams are sys.stdout, "
          "sys.stderr, tf.logging.info, "
          "tf.logging.warning, tf.logging.error. " +
          "File needs to be in the form of 'file://<filepath>'.")

  # If we are only printing a single string scalar, there is no need to format
  if (len(inputs) == 1 and tensor_util.is_tensor(inputs[0])
      and (not isinstance(inputs[0], sparse_tensor.SparseTensor))
      and (inputs[0].shape.ndims == 0)and (inputs[0].dtype == dtypes.string)):
    formatted_string = inputs[0]
  # Otherwise, we construct an appropriate template for the tensors we are
  # printing, and format the template using those tensors.
  else:
    # For each input to this print function, we extract any nested tensors,
    # and construct an appropriate template to format representing the
    # printed input.
    templates = []
    tensors = []
    tensor_free_structure = nest.map_structure(
        lambda x: "" if tensor_util.is_tensor(x) else x,
        inputs)
    tensor_free_template = " ".join(pprint.pformat(x)
                                    for x in tensor_free_structure)
    placeholder = _generate_placeholder_string(tensor_free_template)

    for input_ in inputs:
      placeholders = []
      # Use the nest utilities to flatten & process any nested elements in this
      # input. The placeholder for a tensor in the template should be the
      # placeholder string, and the placeholder for a non-tensor can just be
      # the printed value of the non-tensor itself.
      for x in nest.flatten(input_):
        # support sparse tensors
        if isinstance(x, sparse_tensor.SparseTensor):
          tensors.extend([x.indices, x.values, x.dense_shape])
          placeholders.append(
              "SparseTensor(indices={}, values={}, shape={})".format(
                  placeholder, placeholder, placeholder)
          )
        elif tensor_util.is_tensor(x):
          tensors.append(x)
          placeholders.append(placeholder)
        else:
          placeholders.append(x)

      if isinstance(input_, six.string_types):
        # If the current input to format/print is a normal string, that string
        # can act as the template.
        cur_template = input_
      else:
        # We pack the placeholders into a data structure that matches the
        # input data structure format, then format that data structure
        # into a string template.
        #
        # NOTE: We must use pprint.pformat here for building the template for
        # unordered data structures such as `dict`, because `str` doesn't
        # guarantee orderings, while pprint prints in sorted order. pprint
        # will match the ordering of `nest.flatten`.
        # This even works when nest.flatten reorders OrderedDicts, because
        # pprint is printing *after* the OrderedDicts have been reordered.
        cur_template = pprint.pformat(
            nest.pack_sequence_as(input_, placeholders))
      templates.append(cur_template)

    # We join the templates for the various inputs into a single larger
    # template. We also remove all quotes surrounding the placeholders, so that
    # the formatted/printed output will not contain quotes around tensors.
    # (example of where these quotes might appear: if we have added a
    # placeholder string into a list, then pretty-formatted that list)
    template = " ".join(templates)
    template = template.replace("'" + placeholder + "'", placeholder)
    formatted_string = string_ops.string_format(
        inputs=tensors, template=template, placeholder=placeholder,
        summarize=summarize,
        name=format_name)

  return gen_logging_ops.print_v2(formatted_string,
                                  output_stream=output_stream_string,
                                  name=name)
# pylint: enable=g-doc-args


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
  migrate, look ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

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
    val = gen_logging_ops.histogram_summary(
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
  migrate, look ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

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
    val = gen_logging_ops.image_summary(
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
  migrate, look ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

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
    val = gen_logging_ops.audio_summary_v2(
        tag=tag,
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
    val = gen_logging_ops.merge_summary(inputs=inputs, name=name)
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
  migrate, look ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

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
    val = gen_logging_ops.scalar_summary(tags=tags, values=values, name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val

ops.NotDifferentiable("HistogramSummary")
ops.NotDifferentiable("ImageSummary")
ops.NotDifferentiable("AudioSummary")
ops.NotDifferentiable("AudioSummaryV2")
ops.NotDifferentiable("MergeSummary")
ops.NotDifferentiable("ScalarSummary")
ops.NotDifferentiable("TensorSummary")
ops.NotDifferentiable("TensorSummaryV2")
ops.NotDifferentiable("Timestamp")
