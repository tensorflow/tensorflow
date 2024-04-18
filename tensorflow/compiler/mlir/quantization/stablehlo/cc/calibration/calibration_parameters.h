/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_CALIBRATION_PARAMETERS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_CALIBRATION_PARAMETERS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace stablehlo::quantization {

// TODO: b/321158562 - Make the number of bins configurable.
// Default number of histogram bins for each batch sample.
constexpr int32_t kDefaultNumOfBins = 1 << 9;

// Calculates the bin width from the range and expected number of bins. The
// bin width is formalized to the form of 2^n. As a consequence, the actual
// number of bins might be smaller than the given `num_bins`.
inline float CalculateBinWidth(const float min_value, const float max_value,
                               const int32_t num_bins) {
  const float raw_bin_width = (max_value - min_value) / num_bins;
  return std::pow(2, std::ceil(std::log2(raw_bin_width)));
}

// Calculates the lower bound of the histogram. The lower bound is in form of
// `N * bin_width`.
inline float CalculateLowerBound(const float min_value, const float bin_width) {
  return std::floor(min_value / bin_width) * bin_width;
}

// Calculates the bin index of the current value.
inline int32_t CalculateBinIndex(const float value, const float lower_bound,
                                 const float bin_width) {
  return std::floor((value - lower_bound) / bin_width);
}

// Same as `CalculateBinIndex` but clamps to avoid out-of-bound.
inline int32_t CalculateBinIndexSafe(const float value, const float lower_bound,
                                     const float bin_width,
                                     const int32_t num_bins) {
  const int32_t bin_index = CalculateBinIndex(value, lower_bound, bin_width);
  return std::clamp(bin_index, 0, num_bins - 1);
}

// Checks if the given method is a histogram-based calibration method.
inline bool IsHistogramCalibration(
    const CalibrationOptions::CalibrationMethod method) {
  return method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE ||
         method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE ||
         method == CalibrationOptions::
                       CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY ||
         method ==
             CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC;
}

// Gets the number of bins for the given calibration method.
inline int32_t GetNumBins(const CalibrationOptions::CalibrationMethod method) {
  return IsHistogramCalibration(method) ? kDefaultNumOfBins : 0;
}

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_CALIBRATION_PARAMETERS_H_
