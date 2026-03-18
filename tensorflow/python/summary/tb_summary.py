# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Re-exports the APIs of TF2 summary that live in TensorBoard."""

import functools
import threading

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_audio_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.util.tf_export import tf_export

DEFAULT_BUCKET_COUNT = 30


class LazyTensorCreator:
  """Lazy auto-converting wrapper for a callable that returns a `tf.Tensor`.

  This class wraps an arbitrary callable that returns a `Tensor` so that it
  will be automatically converted to a `Tensor` by any logic that calls
  `tf.convert_to_tensor()`. This also memoizes the callable so that it is
  called at most once.

  The intended use of this class is to defer the construction of a `Tensor`
  (e.g. to avoid unnecessary wasted computation, or ensure any new ops are
  created in a context only available later on in execution), while remaining
  compatible with APIs that expect to be given an already materialized value
  that can be converted to a `Tensor`.

  This class is thread-safe.
  """

  def __init__(self, tensor_callable):
    """Initializes a LazyTensorCreator object.

    Args:
      tensor_callable: A callable that returns a `tf.Tensor`.
    """
    if not callable(tensor_callable):
      raise ValueError("Not a callable: %r" % tensor_callable)
    self._tensor_callable = tensor_callable
    self._tensor = None
    self._tensor_lock = threading.RLock()
    _register_conversion_function_once()

  def __call__(self):
    if self._tensor is None:
      with self._tensor_lock:
        if self._tensor is None:
          self._tensor = self._tensor_callable()
    return self._tensor


def _lazy_tensor_creator_converter(value, dtype=None, name=None, as_ref=False):
  """Converts a LazyTensorCreator to a Tensor for tf.convert_to_tensor."""
  del name  # ignored
  if not isinstance(value, LazyTensorCreator):
    raise RuntimeError("Expected LazyTensorCreator, got %r" % value)
  if as_ref:
    raise RuntimeError("Cannot use LazyTensorCreator to create ref tensor")
  tensor = value()
  if dtype not in (None, tensor.dtype):
    raise RuntimeError(
        "Cannot convert LazyTensorCreator returning dtype %s to dtype %s"
        % (tensor.dtype, dtype)
    )
  return tensor


# Use module-level bit and lock to ensure that registration of the
# LazyTensorCreator conversion function happens only once.
_conversion_registered = False
_conversion_registered_lock = threading.Lock()


def _register_conversion_function_once():
  """Performs one-time registration of `_lazy_tensor_creator_converter`.

  This helper can be invoked multiple times but only registers the conversion
  function on the first invocation, making it suitable for calling when
  constructing a LazyTensorCreator.

  Deferring the registration is necessary because doing it at at module import
  time would trigger the lazy TensorFlow import to resolve, and that in turn
  would break the delicate `tf.summary` import cycle avoidance scheme.
  """
  global _conversion_registered
  if not _conversion_registered:
    with _conversion_registered_lock:
      if not _conversion_registered:
        try:
          ops.register_tensor_conversion_function(
              base_type=LazyTensorCreator,
              conversion_func=_lazy_tensor_creator_converter,
              priority=0,
          )
        except AttributeError:
          pass
        _conversion_registered = True


def _create_summary_metadata(description, plugin_name):
  return summary_pb2.SummaryMetadata(
      summary_description=description,
      plugin_data=summary_pb2.SummaryMetadata.PluginData(
          plugin_name=plugin_name, content=b""
      ),
  )


@tf_export("summary.audio", v1=[])
def audio(
    name,
    data,
    sample_rate,
    step=None,
    max_outputs=3,
    encoding=None,
    description=None,
):
  """Write an audio summary.

  Arguments:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A `Tensor` representing audio data with shape `[k, t, c]`, where `k`
      is the number of audio clips, `t` is the number of frames, and `c` is the
      number of channels. Elements should be floating-point values in `[-1.0,
      1.0]`. Any of the dimensions may be statically unknown (i.e., `None`).
    sample_rate: An `int` or rank-0 `int32` `Tensor` that represents the sample
      rate, in Hz. Must be positive.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this many
      audio clips will be emitted at each step. When more than `max_outputs`
      many clips are provided, the first `max_outputs` many clips will be used
      and the rest silently discarded.
    encoding: Optional constant `str` for the desired encoding. Only "wav" is
      currently supported, but this is not guaranteed to remain the default, so
      if you want "wav" in particular, set this explicitly.
    description: Optional long-form description for this summary, as a constant
      `str`. Markdown is supported. Defaults to empty.

  Returns:
    True on success, or false if no summary was emitted because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  if encoding is None:
    encoding = "wav"
  if encoding != "wav":
    raise ValueError("Unknown encoding: %r" % encoding)
  summary_metadata = _create_summary_metadata(description, "audio")
  inputs = [data, sample_rate, max_outputs, step]
  # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
  summary_scope = (
      getattr(summary_ops_v2, "summary_scope", None)
  )
  with summary_scope(name, "audio_summary", values=inputs) as (tag, _):
    # Defer audio encoding preprocessing by passing it as a callable to write(),
    # wrapped in a LazyTensorCreator for backwards compatibility, so that we
    # only do this work when summaries are actually written.
    @LazyTensorCreator
    def lazy_tensor():
      check_ops.assert_rank(data, 3)
      check_ops.assert_non_negative(max_outputs)
      limited_audio = data[:max_outputs]
      encode_fn = functools.partial(
          gen_audio_ops.encode_wav, sample_rate=sample_rate
      )
      encoded_audio = map_fn_lib.map_fn(
          encode_fn,
          limited_audio,
          dtype=dtypes.string,
          name="encode_each_audio",
      )
      # Workaround for map_fn returning float dtype for an empty elems input.
      encoded_audio = cond.cond(
          array_ops.shape(input=encoded_audio)[0] > 0,
          lambda: encoded_audio,
          lambda: constant_op.constant(
              [], dtypes.string
          ),
      )
      limited_labels = array_ops.tile(
          [""], array_ops.shape(input=limited_audio)[:1]
      )
      return array_ops.transpose(
          a=array_ops_stack.stack(
              [encoded_audio, limited_labels]
          )
      )

    # To ensure that audio encoding logic is only executed when summaries
    # are written, we pass callable to `tensor` parameter.
    return summary_ops_v2.write(
        tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
    )


@tf_export("summary.histogram", v1=[])
def histogram(name, data, step=None, buckets=None, description=None):
  """Write a histogram summary.

  See also `tf.summary.scalar`, `tf.summary.SummaryWriter`.

  Writes a histogram to the current default summary writer, for later analysis
  in TensorBoard's 'Histograms' and 'Distributions' dashboards (data written
  using this API will appear in both places). Like `tf.summary.scalar` points,
  each histogram is associated with a `step` and a `name`. All the histograms
  with the same `name` constitute a time series of histograms.

  The histogram is calculated over all the elements of the given `Tensor`
  without regard to its shape or rank.

  This example writes 2 histograms:

  ```python
  w = tf.summary.create_file_writer('test/logs')
  with w.as_default():
      tf.summary.histogram("activations", tf.random.uniform([100, 50]), step=0)
      tf.summary.histogram("initial_weights", tf.random.normal([1000]), step=0)
  ```

  A common use case is to examine the changing activation patterns (or lack
  thereof) at specific layers in a neural network, over time.

  ```python
  w = tf.summary.create_file_writer('test/logs')
  with w.as_default():
  for step in range(100):
      # Generate fake "activations".
      activations = [
          tf.random.normal([1000], mean=step, stddev=1),
          tf.random.normal([1000], mean=step, stddev=10),
          tf.random.normal([1000], mean=step, stddev=100),
      ]

      tf.summary.histogram("layer1/activate", activations[0], step=step)
      tf.summary.histogram("layer2/activate", activations[1], step=step)
      tf.summary.histogram("layer3/activate", activations[2], step=step)
  ```

  Arguments:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A `Tensor` of any shape. The histogram is computed over its elements,
      which must be castable to `float64`.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    buckets: Optional positive `int`. The output will have this many buckets,
      except in two edge cases. If there is no data, then there are no buckets.
      If there is data but all points have the same value, then all buckets'
      left and right endpoints are the same and only the last bucket has nonzero
      count. Defaults to 30 if not specified.
    description: Optional long-form description for this summary, as a constant
      `str`. Markdown is supported. Defaults to empty.

  Returns:
    True on success, or false if no summary was emitted because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  # Avoid building unused gradient graphs for conds below. This works around
  # an error building second-order gradient graphs when XlaDynamicUpdateSlice
  # is used, and will generally speed up graph building slightly.
  data = array_ops.stop_gradient(data)
  summary_metadata = _create_summary_metadata(description, "histograms")
  # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
  summary_scope = (
      getattr(summary_ops_v2, "summary_scope", None)
  )

  # TODO(ytjing): add special case handling.
  with summary_scope(
      name, "histogram_summary", values=[data, buckets, step]
  ) as (tag, _):
    # Defer histogram bucketing logic by passing it as a callable to
    # write(), wrapped in a LazyTensorCreator for backwards
    # compatibility, so that we only do this work when summaries are
    # actually written.
    @LazyTensorCreator
    def lazy_tensor():
      return _buckets(data, buckets)

    return summary_ops_v2.write(
        tag=tag,
        tensor=lazy_tensor,
        step=step,
        metadata=summary_metadata,
    )


def _buckets(data, bucket_count=None):
  """Create a TensorFlow op to group data into histogram buckets.

  Arguments:
    data: A `Tensor` of any shape. Must be castable to `float64`.
    bucket_count: Optional non-negative `int` or scalar `int32` `Tensor`,
      defaults to 30.

  Returns:
    A `Tensor` of shape `[k, 3]` and type `float64`. The `i`th row is
    a triple `[left_edge, right_edge, count]` for a single bucket.
    The value of `k` is either `bucket_count` or `0` (when input data
    is empty).
  """
  if bucket_count is None:
    bucket_count = DEFAULT_BUCKET_COUNT
  with ops.name_scope("buckets"):
    check_ops.assert_scalar(bucket_count)
    check_ops.assert_type(
        bucket_count, dtypes.int32
    )
    # Treat a negative bucket count as zero.
    bucket_count = math_ops.maximum(
        0, bucket_count
    )
    data = array_ops.reshape(data, shape=[-1])  # flatten
    data = math_ops.cast(
        data, dtypes.float64
    )
    data_size = array_ops.size(input=data)
    is_empty = math_ops.logical_or(
        math_ops.equal(data_size, 0),
        math_ops.less_equal(bucket_count, 0),
    )

    def when_empty():
      """When input data is empty or bucket_count is zero.

      1. If bucket_count is specified as zero, an empty tensor of shape
        (0, 3) will be returned.
      2. If the input data is empty, a tensor of shape (bucket_count, 3)
        of all zero values will be returned.
      """
      return array_ops.zeros(
          (bucket_count, 3), dtype=dtypes.float64
      )

    def when_nonempty():
      min_ = math_ops.reduce_min(input_tensor=data)
      max_ = math_ops.reduce_max(input_tensor=data)
      range_ = max_ - min_
      has_single_value = math_ops.equal(range_, 0)

      def when_multiple_values():
        """When input data contains multiple values."""
        bucket_width = range_ / math_ops.cast(
            bucket_count, dtypes.float64
        )
        offsets = data - min_
        bucket_indices = math_ops.cast(
            math_ops.floor(offsets / bucket_width),
            dtype=dtypes.int32,
        )
        clamped_indices = math_ops.minimum(
            bucket_indices, bucket_count - 1
        )
        # Use float64 instead of float32 to avoid accumulating floating point
        # error later in tf.reduce_sum when summing more than 2^24
        # individual `1.0` values. See
        # https://github.com/tensorflow/tensorflow/issues/51419 for details.
        one_hots = array_ops.one_hot(
            clamped_indices,
            depth=bucket_count,
            dtype=dtypes.float64,
        )
        bucket_counts = math_ops.cast(
            math_ops.reduce_sum(
                input_tensor=one_hots, axis=0
            ),
            dtype=dtypes.float64,
        )
        edges = math_ops.linspace(min_, max_, bucket_count + 1)
        # Ensure edges[-1] == max_, which TF's linspace implementation does not
        # do, leaving it subject to the whim of floating point rounding error.
        edges = array_ops.concat([edges[:-1], [max_]], 0)
        left_edges = edges[:-1]
        right_edges = edges[1:]
        return array_ops.transpose(
            a=array_ops_stack.stack(
                [left_edges, right_edges, bucket_counts]
            )
        )

      def when_single_value():
        """When input data contains a single unique value."""
        # Left and right edges are the same for single value input.
        edges = array_ops.fill([bucket_count], max_)
        # Bucket counts are 0 except the last bucket (if bucket_count > 0),
        # which is `data_size`. Ensure that the resulting counts vector has
        # length `bucket_count` always, including the bucket_count==0 case.
        zeroes = array_ops.fill([bucket_count], 0)
        bucket_counts = math_ops.cast(
            array_ops.concat(
                [zeroes[:-1], [data_size]], 0
            )[:bucket_count],
            dtype=dtypes.float64,
        )
        return array_ops.transpose(
            a=array_ops_stack.stack(
                [edges, edges, bucket_counts]
            )
        )

      return cond.cond(
          has_single_value, when_single_value, when_multiple_values
      )

    return cond.cond(
        is_empty, when_empty, when_nonempty
    )


@tf_export("summary.image", v1=[])
def image(name, data, step=None, max_outputs=3, description=None):
  """Write an image summary.

  See also `tf.summary.scalar`, `tf.summary.SummaryWriter`.

  Writes a collection of images to the current default summary writer. Data
  appears in TensorBoard's 'Images' dashboard. Like `tf.summary.scalar` points,
  each collection of images is associated with a `step` and a `name`.  All the
  image collections with the same `name` constitute a time series of image
  collections.

  This example writes 2 random grayscale images:

  ```python
  w = tf.summary.create_file_writer('test/logs')
  with w.as_default():
    image1 = tf.random.uniform(shape=[8, 8, 1])
    image2 = tf.random.uniform(shape=[8, 8, 1])
    tf.summary.image("grayscale_noise", [image1, image2], step=0)
  ```

  To avoid clipping, data should be converted to one of the following:

  - floating point values in the range [0,1], or
  - uint8 values in the range [0,255]

  ```python
  # Convert the original dtype=int32 `Tensor` into `dtype=float64`.
  rgb_image_float = tf.constant([
    [[1000, 0, 0], [0, 500, 1000]],
  ]) / 1000
  tf.summary.image("picture", [rgb_image_float], step=0)

  # Convert original dtype=uint8 `Tensor` into proper range.
  rgb_image_uint8 = tf.constant([
    [[1, 1, 0], [0, 0, 1]],
  ], dtype=tf.uint8) * 255
  tf.summary.image("picture", [rgb_image_uint8], step=1)
  ```

  Arguments:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A `Tensor` representing pixel data with shape `[k, h, w, c]`, where
      `k` is the number of images, `h` and `w` are the height and width of the
      images, and `c` is the number of channels, which should be 1, 2, 3, or 4
      (grayscale, grayscale with alpha, RGB, RGBA). Any of the dimensions may be
      statically unknown (i.e., `None`). Floating point data will be clipped to
      the range [0,1]. Other data types will be clipped into an allowed range
      for safe casting to uint8, using `tf.image.convert_image_dtype`.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this many
      images will be emitted at each step. When more than `max_outputs` many
      images are provided, the first `max_outputs` many images will be used and
      the rest silently discarded.
    description: Optional long-form description for this summary, as a constant
      `str`. Markdown is supported. Defaults to empty.

  Returns:
    True on success, or false if no summary was emitted because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = _create_summary_metadata(description, "images")
  # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
  summary_scope = (
      getattr(summary_ops_v2, "summary_scope", None)
  )
  with summary_scope(
      name, "image_summary", values=[data, max_outputs, step]
  ) as (tag, _):
    # Defer image encoding preprocessing by passing it as a callable to write(),
    # wrapped in a LazyTensorCreator for backwards compatibility, so that we
    # only do this work when summaries are actually written.
    @LazyTensorCreator
    def lazy_tensor():
      check_ops.assert_rank(data, 4)
      check_ops.assert_non_negative(
          max_outputs
      )
      images = image_ops.convert_image_dtype(
          data, dtypes.uint8, saturate=True
      )
      limited_images = images[:max_outputs]
      encoded_images = image_ops.encode_png(
          limited_images
      )
      image_shape = array_ops.shape(input=images)
      dimensions = array_ops_stack.stack(
          [
              string_ops.as_string(
                  image_shape[2], name="width"
              ),
              string_ops.as_string(
                  image_shape[1], name="height"
              ),
          ],
          name="dimensions",
      )
      return array_ops.concat(
          [dimensions, encoded_images], axis=0
      )

    # To ensure that image encoding logic is only executed when summaries
    # are written, we pass callable to `tensor` parameter.
    return summary_ops_v2.write(
        tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
    )


@tf_export("summary.scalar", v1=[])
def scalar(name, data, step=None, description=None):
  """Write a scalar summary.

  See also `tf.summary.image`, `tf.summary.histogram`,
  `tf.summary.SummaryWriter`.

  Writes simple numeric values for later analysis in TensorBoard.  Writes go to
  the current default summary writer. Each summary point is associated with an
  integral `step` value. This enables the incremental logging of time series
  data.  A common usage of this API is to log loss during training to produce
  a loss curve.

  For example:

  ```python
  test_summary_writer = tf.summary.create_file_writer('test/logdir')
  with test_summary_writer.as_default():
      tf.summary.scalar('loss', 0.345, step=1)
      tf.summary.scalar('loss', 0.234, step=2)
      tf.summary.scalar('loss', 0.123, step=3)
  ```

  Multiple independent time series may be logged by giving each series a unique
  `name` value.

  See [Get started with
  TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
  for more examples of effective usage of `tf.summary.scalar`.

  In general, this API expects that data points are logged with a monotonically
  increasing step value. Duplicate points for a single step or points logged out
  of order by step are not guaranteed to display as desired in TensorBoard.

  Arguments:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A real numeric scalar value, convertible to a `float32` Tensor.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    description: Optional long-form description for this summary, as a constant
      `str`. Markdown is supported. Defaults to empty.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = _create_summary_metadata(description, "scalars")
  # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
  summary_scope = (
      getattr(summary_ops_v2, "summary_scope", None)
  )
  with summary_scope(name, "scalar_summary", values=[data, step]) as (tag, _):
    check_ops.assert_scalar(data)
    return summary_ops_v2.write(
        tag=tag,
        tensor=math_ops.cast(
            data, dtypes.float32
        ),
        step=step,
        metadata=summary_metadata,
    )


@tf_export("summary.text", v1=[])
def text(name, data, step=None, description=None):
  r"""Write a text summary.

  See also `tf.summary.scalar`, `tf.summary.SummaryWriter`, `tf.summary.image`.

  Writes text Tensor values for later visualization and analysis in TensorBoard.
  Writes go to the current default summary writer.  Like `tf.summary.scalar`
  points, text points are each associated with a `step` and a `name`.
  All the points with the same `name` constitute a time series of text values.

  For Example:
  ```python
  test_summary_writer = tf.summary.create_file_writer('test/logdir')
  with test_summary_writer.as_default():
      tf.summary.text('first_text', 'hello world!', step=0)
      tf.summary.text('first_text', 'nice to meet you!', step=1)
  ```

  The text summary can also contain Markdown, and TensorBoard will render the
  text
  as such.

  ```python
  with test_summary_writer.as_default():
      text_data = '''
            | *hello* | *there* |
            |---------|---------|
            | this    | is      |
            | a       | table   |
      '''
      text_data = '\n'.join(l.strip() for l in text_data.splitlines())
      tf.summary.text('markdown_text', text_data, step=0)
  ```

  Since text is Tensor valued, each text point may be a Tensor of string values.
  rank-1 and rank-2 Tensors are rendered as tables in TensorBoard.  For higher
  ranked
  Tensors, you'll see just a 2D slice of the data.  To avoid this, reshape the
  Tensor
  to at most rank-2 prior to passing it to this function.

  Demo notebook at
  ["Displaying text data in
  TensorBoard"](https://www.tensorflow.org/tensorboard/text_summaries).

  Arguments:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A UTF-8 string Tensor value.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    description: Optional long-form description for this summary, as a constant
      `str`. Markdown is supported. Defaults to empty.

  Returns:
    True on success, or false if no summary was emitted because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = _create_summary_metadata(description, "text")
  # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
  summary_scope = (
      getattr(summary_ops_v2, "summary_scope", None)
  )
  with summary_scope(name, "text_summary", values=[data, step]) as (tag, _):
    check_ops.assert_type(
        data, dtypes.string
    )
    return summary_ops_v2.write(
        tag=tag, tensor=data, step=step, metadata=summary_metadata
    )



