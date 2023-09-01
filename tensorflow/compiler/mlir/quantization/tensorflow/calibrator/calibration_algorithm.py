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
    # i-th bin has a range [bins[i], bins[i + 1]).
    # bins[i] = lower_bound + i * bin_width
    # bins[i + 1] = lower_bound + (i + 1) * bin_width
    # So hist_mids[i] = (lower_bound + bin_width / 2) + bin_width * i
    first_mid = self._lower_bound + self._bin_width / 2
    last_mid = first_mid + (self._num_bins - 1) * self._bin_width
    self._hist_mids = np.linspace(first_mid, last_mid, self._num_bins)


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
