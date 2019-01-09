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
"""mel conversion ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util.tf_export import tf_export


# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _mel_to_hertz(mel_values, name=None):
  """Converts frequencies in `mel_values` from the mel scale to linear scale.

  Args:
    mel_values: A `Tensor` of frequencies in the mel scale.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type as `mel_values` containing linear
    scale frequencies in Hertz.
  """
  with ops.name_scope(name, 'mel_to_hertz', [mel_values]):
    mel_values = ops.convert_to_tensor(mel_values)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        math_ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def _hertz_to_mel(frequencies_hertz, name=None):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.

  Args:
    frequencies_hertz: A `Tensor` of frequencies in Hertz.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
  """
  with ops.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
    frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
    return _MEL_HIGH_FREQUENCY_Q * math_ops.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _validate_arguments(num_mel_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype):
  """Checks the inputs to linear_to_mel_weight_matrix."""
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if lower_edge_hertz < 0.0:
    raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                     lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                     % (upper_edge_hertz, sample_rate))
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Got: %s' % dtype)


@tf_export('signal.linear_to_mel_weight_matrix')
def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0,
                                dtype=dtypes.float32,
                                name=None):
  """Returns a matrix to warp linear scale spectrograms to the
  [mel scale](https://en.wikipedia.org/wiki/Mel_scale).

  Returns a weight matrix that can be used to re-weight a `Tensor` containing
  `num_spectrogram_bins` linearly sampled frequency information from
  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
  `[lower_edge_hertz, upper_edge_hertz]` on the
  [mel scale](https://en.wikipedia.org/wiki/Mel_scale).

  For example, the returned matrix `A` can be used to right-multiply a
  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
  scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"
  `M` of shape `[frames, num_mel_bins]`.

      # `S` has shape [frames, num_spectrogram_bins]
      # `M` has shape [frames, num_mel_bins]
      M = tf.matmul(S, A)

  The matrix can be used with `tf.tensordot` to convert an arbitrary rank
  `Tensor` of linear-scale spectral bins into the mel scale.

      # S has shape [..., num_spectrogram_bins].
      # M has shape [..., num_mel_bins].
      M = tf.tensordot(S, A, 1)
      # tf.tensordot does not support shape inference for this case yet.
      M.set_shape(S.shape[:-1].concatenate(A.shape[-1:]))

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: Python float. Samples per second of the input signal used to
      create the spectrogram. We need this to figure out the actual frequencies
      for each spectrogram bin, which dictates how they are mapped into the mel
      scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.

  Raises:
    ValueError: If num_mel_bins/num_spectrogram_bins/sample_rate are not
      positive, lower_edge_hertz is negative, frequency edges are incorrectly
      ordered, or upper_edge_hertz is larger than the Nyquist frequency.
  """
  with ops.name_scope(name, 'linear_to_mel_weight_matrix') as name:
    # Note: As num_spectrogram_bins is passed to `math_ops.linspace`
    # and the validation is already done in linspace (both in shape function
    # and in kernel), there is no need to validate num_spectrogram_bins here.
    _validate_arguments(num_mel_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype)

    # This function can be constant folded by graph optimization since there are
    # no Tensor inputs.
    sample_rate = ops.convert_to_tensor(
        sample_rate, dtype, name='sample_rate')
    lower_edge_hertz = ops.convert_to_tensor(
        lower_edge_hertz, dtype, name='lower_edge_hertz')
    upper_edge_hertz = ops.convert_to_tensor(
        upper_edge_hertz, dtype, name='upper_edge_hertz')
    zero = ops.convert_to_tensor(0.0, dtype)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = math_ops.linspace(
        zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = array_ops.expand_dims(
        _hertz_to_mel(linear_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = shape_ops.frame(
        math_ops.linspace(_hertz_to_mel(lower_edge_hertz),
                          _hertz_to_mel(upper_edge_hertz),
                          num_mel_bins + 2), frame_length=3, frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(array_ops.reshape(
        t, [1, num_mel_bins]) for t in array_ops.split(
            band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = math_ops.maximum(
        zero, math_ops.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    return array_ops.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name)
