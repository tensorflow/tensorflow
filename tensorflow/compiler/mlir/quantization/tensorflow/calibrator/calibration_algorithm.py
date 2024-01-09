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
"""Defines CalibrationAlgorithm for calculating min and max values calculated by calibration method."""
import abc
import itertools
import logging

import numpy as np

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2


_CalibrationMethod = quant_opts_pb2.CalibrationOptions.CalibrationMethod
_REGISTRY = {}


def _implements(calib_method: _CalibrationMethod):
  def decorator(cls):
    assert calib_method not in _REGISTRY
    _REGISTRY[calib_method] = cls
    return cls

  return decorator


class _CalibrationAlgorithmBase(abc.ABC):
  """Abstract base class for calibration algorithm."""

  def __init__(
      self,
      statistics: calib_stats_pb2.CalibrationStatistics,
      calib_opts: quant_opts_pb2.CalibrationOptions,
  ):
    self._statistics = statistics
    self._calib_opts = calib_opts

  @abc.abstractmethod
  def get_min_max_value(self) -> tuple[float, float]:
    pass


class _HistogramCalibrationAlgorithmBase(_CalibrationAlgorithmBase):
  """Base class for histogram calibrators."""

  def __init__(
      self,
      statistics: calib_stats_pb2.CalibrationStatistics,
      calib_opts: quant_opts_pb2.CalibrationOptions,
  ):
    """Builds histogram using statistics.histogram_statistics.

    lower_bound                                    hist_mid
         v                                            v
         |=========|=========|=========|=========|=========|
                    bin width

    Args:
      statistics: Collected calibration statistics.
      calib_opts: Calibration options used for calculating min and max.
    """
    super().__init__(statistics, calib_opts)
    hist_stats = statistics.histogram_statistics
    self._bin_width = hist_stats.bin_width
    self._lower_bound = hist_stats.lower_bound
    self._hist_freq = np.array(hist_stats.hist_freq)
    self._num_bins = len(self._hist_freq)
    self._num_bits = 8
    # i-th bin has a range [bins[i], bins[i + 1]).
    # bins[i] = lower_bound + i * bin_width
    # bins[i + 1] = lower_bound + (i + 1) * bin_width
    # So hist_mids[i] = (lower_bound + bin_width / 2) + bin_width * i
    first_mid = self._lower_bound + self._bin_width / 2
    last_mid = first_mid + (self._num_bins - 1) * self._bin_width
    self._hist_mids = np.linspace(first_mid, last_mid, self._num_bins)

  def _get_dequantized_hist_mids_after_quantize(
      self, quant_min: float, quant_max: float
  ) -> np.ndarray:
    """Quantizes and dequantizes hist_mids using quant_min and quant_max.

    Quantization converts the range of numbers from [quant_min, quant_max] to
    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and
    values greater than quant_max are converted to 2^num_bits - 1.

    The histogram represents the distribution of the data, and our goal is to
    find the quant_min and quant_max that best describe this distribution. To do
    this, we quantize hist_mids using quant_min and quant_max and dequantize
    them again. Then the difference between hist_mids and dequantized hist_mids
    equates to quantization error when using quant_min and quant_max.


    Args:
      quant_min: The minimum real value that can be represented by a quantized
        value.
      quant_max: The maximum real value that can be represented by a quantized
        value.

    Returns:
      dequantized hist_mids after quantizing by quant_min and quant_max
    """
    maxbound = 2**self._num_bits - 1
    minbound = 0
    scale = (quant_max - quant_min) / maxbound
    zero_point = -quant_min / scale

    # Limit the range of zero_point and scale in case (quant_max - quant_min)
    # is unusually small.
    if abs(zero_point) > 9e9:
      zero_point = 9e9
    if abs(scale) < 1e-9:
      scale = 1e-9

    zero_point = round(zero_point)
    quantized_hist_mids = np.clip(
        np.round(self._hist_mids / scale) + zero_point, minbound, maxbound
    )
    dequantized_hist_mids = scale * (quantized_hist_mids - zero_point)
    return dequantized_hist_mids

  def _get_weighted_mean_squared_error(
      self, quant_min, quant_max
  ) -> tuple[float, float, float]:
    """Gets mean squared error between hist_mids and dequantized hist_mids.

    Quantization converts the range of numbers from [quant_min, quant_max] to
    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and
    values greater than quant_max are converted to 2^num_bits - 1.

    Args:
      quant_min: The minimum real value that can be represented by a quantized
        value.
      quant_max: The maximum real value that can be represented by a quantized
        value.

    Returns:
      (error, quant_min, quant_max): Tuple of weighted mean squared error.
      error = (hist_mids - dequantized_hist_mids)**2 * hist_freq
    """
    dequantized_hist_mids = self._get_dequantized_hist_mids_after_quantize(
        quant_min, quant_max
    )
    squared_error = (self._hist_mids - dequantized_hist_mids) ** 2
    weighted_error = np.sum(squared_error * self._hist_freq)
    return (weighted_error, quant_min, quant_max)

  def _get_min_max_value_by_expanding_range(
      self, start_idx: int
  ) -> tuple[float, float]:
    """Starting from start_idx, expand left and right alternately to find the min value of mse loss.

    Args:
      start_idx: Index to start quantization.

    Returns:
      (min_value, max_value): Min and max calculated.
    """
    # Tuple of (mse_error, quant_min, quant_max).
    mse_min = (float('inf'), float('inf'), float('inf'))
    left, right = start_idx, start_idx

    # If this value is true, it moves left, otherwise it moves right.
    move_left = True
    while not (left == 0 and right == self._num_bins - 1):
      # Decrease left if right can't be moved or move_left is true.
      if (move_left and left > 0) or (right == self._num_bins - 1):
        left = max(left - 1, 0)
      # Else increase right.
      else:
        right = min(right + 1, self._num_bins - 1)
      # Toogle the move_left.
      move_left = not move_left
      quant_min, quant_max = self._hist_mids[left], self._hist_mids[right]
      mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
      mse_min = min(mse_tuple, mse_min)
    # Extract (quant_min, quant_max) from (mse_error, quant_min, quant_max).
    min_value, max_value = mse_min[1], mse_min[2]
    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX)
class _MinMax(_CalibrationAlgorithmBase):
  """MinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  MinMax calibration calculates the global min and global max values.

  global min = min of given sample inputs
  global max = max of given sample inputs
  """

  def get_min_max_value(self) -> tuple[float, float]:
    """Calculates the global min and max values.

    Returns:
      (min_value, max_value): Min and max calculated using MinMax
    """
    return (
        self._statistics.min_max_statistics.global_min,
        self._statistics.min_max_statistics.global_max,
    )


@_implements(_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX)
class _AverageMinMax(_CalibrationAlgorithmBase):
  """AverageMinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  AverageMinMax calibration calculates the average of min and max values.
  average of min = sum of min values / number of samples
  average of max = sum of max values / number of samples
  """

  def get_min_max_value(self) -> tuple[float, float]:
    """Calculates the average of min and max values.

    Returns:
      (min_value, max_value): Min and max calculated using AverageMinMax

    Raises:
      ValueError: num_samples is 0.
    """
    average_min_max_statistics = self._statistics.average_min_max_statistics
    # num_samples is guaranteed to be larger than 0 because
    # get_statistics_from_calibrator throws an exception if num_samples == 0.
    num_samples = average_min_max_statistics.num_samples
    if num_samples == 0:
      raise ValueError(
          'num_samples must not be 0 when calibration method is'
          f' AverageMinMax: {self._calib_opts}'
      )
    min_value, max_value = (
        average_min_max_statistics.min_sum / num_samples,
        average_min_max_statistics.max_sum / num_samples,
    )

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE)
class _HistogramPercentile(_HistogramCalibrationAlgorithmBase):
  """HistogramPercentile for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    """Calculates min and max from statistics using calibration options.

    A "percentile" is a statistical concept that represents the value below
    which a given percentage of data falls in a dataset. It involves sorting the
    data from smallest to largest and then finding the value at a specified
    percentage position. For example, the 0.01 percentile represents the value
    in a given data set that corresponds to the lowest 0.01% of the data.

    HistogramPercentile calibration uses min_percentile and max_percentile to
    find min and max.

    min_percentile and max_percentile must be in range [0, 100].
    min_percentile is 0.001 by default.
    max_percentile is 99.999 by default.

    Returns:
      (min_value, max_value): Min and max calculated using HistogramPercentile
    """
    total_freq = sum(self._hist_freq)
    # hist_freq_cumsum is dividing cumulative sum of hist_freq by total_freq
    # hist_freq_cumsum's value is in range [0, 1] by its definition
    hist_freq_cumsum = np.cumsum(self._hist_freq) / total_freq

    # min_percentile and max_percentile are converted from [0, 100] to [0, 1].
    min_quantile, max_quantile = (
        self._calib_opts.calibration_parameters.min_percentile / 100.0,
        self._calib_opts.calibration_parameters.max_percentile / 100.0,
    )

    # Get index of min/max quantile.
    min_quantile_idx, max_quantile_idx = (
        np.searchsorted(hist_freq_cumsum, min_quantile, side='right'),
        np.searchsorted(hist_freq_cumsum, max_quantile, side='left'),
    )

    # Get value of min/max quantile index.
    min_value, max_value = (
        self._hist_mids[min_quantile_idx],
        self._hist_mids[max_quantile_idx],
    )

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE)
class _HistogramMseBruteforce(_HistogramCalibrationAlgorithmBase):
  """HistogramMseBruteforce for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    """Finds the optimal quant_min and quant_max by testing all possible cases.

    It guarantees optimal quant_min and quant_max for the representative
    dataset, but not for the test dataset.

    Returns:
      (min_value, max_value): Min and max calculated using
      HistogramMseBruteforce.
    """
    if self._num_bins > 512:
      logging.warning(
          'num_bins=%d is too large. The HISTOGRAM_MSE_BRUTEFORCE method tests'
          ' all histogram mid value pairs, so it may take a long time.',
          self._num_bins,
      )
    # Tuple of (mse_error, quant_min, quant_max).
    mse_min = (float('inf'), float('inf'), float('inf'))

    # Calculate the error for all hist_mid pairs.
    for left, right in itertools.combinations(range(self._num_bins), 2):
      quant_min, quant_max = self._hist_mids[left], self._hist_mids[right]
      mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
      mse_min = min(mse_tuple, mse_min)
    min_value, max_value = mse_min[1], mse_min[2]

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY)
class _HistogramMseMaxFrequency(_HistogramCalibrationAlgorithmBase):
  """HistogramMseMaxFrequency for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    """Finds min and max starting from the index of the max frequency.

     The HistogramMseMaxFrequency method starts from the bin with the highest
     frequency and expands the range to both sides. This performs well when data
     is well spread on both sides of the max frequency.

    Returns:
      (min_value, max_value): Min and max calculated using method to expand the
      range based on max frequency.
    """
    # Find the index of max frequency.
    freq_max_idx = np.argmax(self._hist_freq)
    return self._get_min_max_value_by_expanding_range(freq_max_idx)


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC)
class _HistogramMseSymmetric(_HistogramCalibrationAlgorithmBase):
  """HistogramMseSymmetric for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    """Finds min and max starting from the center index.

    The HistogramMseSymmetric method starts from the center bin and expands the
    range to both sides. This works better when the data is well-centered.

    Returns:
      (min_value, max_value): Min and max calculated using the method starting
      from center and expanding.
    """

    # This function is currently only called in this method, but will be used in
    # other methods in the future.
    return self._get_min_max_value_by_expanding_range(self._num_bins // 2)


def get_min_max_value(
    statistics: calib_stats_pb2.CalibrationStatistics,
    calib_opts: quant_opts_pb2.CalibrationOptions,
) -> tuple[float, float]:
  """Calculates min and max from statistics using calibration options.

  Args:
    statistics: Collected calibration statistics.
    calib_opts: Calibration options used for calculating min and max.

  Returns:
    (min_value, max_value): Min and max calculated using calib_opts.

  Raises:
    ValueError: Unsupported calibration method is given.
  """
  calib_method = calib_opts.calibration_method
  if calib_method not in _REGISTRY:
    raise ValueError(f'Unsupported calibration method: {calib_method}')

  calibration_algorithm = _REGISTRY[calib_method](statistics, calib_opts)
  return calibration_algorithm.get_min_max_value()
