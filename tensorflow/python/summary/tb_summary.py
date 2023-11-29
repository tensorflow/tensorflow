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

from tensorflow.python.util.tf_export import tf_export

_TENSORBOARD_NOT_INSTALLED_ERROR = (
    "TensorBoard is not installed, missing implementation for"
)


class TBNotInstalledError(Exception):

  def __init__(self, summary_api):
    self.error_message = f"{_TENSORBOARD_NOT_INSTALLED_ERROR} {summary_api}"
    super().__init__(self.error_message)


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
  try:
    from tensorboard.summary.v2 import audio as audio_v2  # pylint: disable=g-import-not-at-top, g-importing-member
  except ImportError as exc:
    raise TBNotInstalledError("tf.summary.audio") from exc
  return audio_v2(
      name=name,
      data=data,
      sample_rate=sample_rate,
      step=step,
      max_outputs=max_outputs,
      encoding=encoding,
      description=description,
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
  try:
    from tensorboard.summary.v2 import histogram as histogram_v2  # pylint: disable=g-import-not-at-top, g-importing-member
  except ImportError as exc:
    raise TBNotInstalledError("tf.summary.histogram") from exc
  return histogram_v2(
      name=name, data=data, step=step, buckets=buckets, description=description
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
  try:
    from tensorboard.summary.v2 import image as image_v2  # pylint: disable=g-import-not-at-top, g-importing-member
  except ImportError as exc:
    raise TBNotInstalledError("tf.summary.image") from exc
  return image_v2(
      name=name,
      data=data,
      step=step,
      max_outputs=max_outputs,
      description=description,
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
  try:
    from tensorboard.summary.v2 import scalar as scalar_v2  # pylint: disable=g-import-not-at-top, g-importing-member
  except ImportError as exc:
    raise TBNotInstalledError("tf.summary.scalar") from exc
  return scalar_v2(name=name, data=data, step=step, description=description)


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
  try:
    from tensorboard.summary.v2 import text as text_v2  # pylint: disable=g-import-not-at-top, g-importing-member
  except ImportError as exc:
    raise TBNotInstalledError("tf.summary.text") from exc
  return text_v2(name=name, data=data, step=step, description=description)
