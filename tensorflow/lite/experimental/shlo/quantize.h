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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZE_H_

#include <algorithm>

#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {

// Converts floating-point values of the expressed type into corresponding
// integer values of the storage type using the zero point and scale associated
// with the quantized element type.
template <typename StorageT, typename ExpressedT>
inline constexpr StorageT Quantize(ExpressedT expressed_value,
                                   StorageT zero_point, ExpressedT scale_inv,
                                   StorageT min_value, StorageT max_value) {
  const ExpressedT rounding_extra =
      (expressed_value > 0) ? ExpressedT(0.5f) : ExpressedT(-0.5f);
  ExpressedT tmp = expressed_value * scale_inv + rounding_extra;

  // Clamp the value in case of overflow/underflow. This is needed to avoid
  // getting a SIGILL exception when casting down below.
  tmp = std::clamp(tmp, static_cast<ExpressedT>(min_value),
                   static_cast<ExpressedT>(max_value));
  auto rounded_value = static_cast<StorageT>(tmp);
  StorageT storage_value(rounded_value + zero_point);

  // Clamp again using the min & max values.
  return std::clamp(storage_value, min_value, max_value);
}

// A DataType dispatched version of Quantize, this allows for leveraging the min
// and max values of the DataType, which may not necessarily be the same as the
// min and max values of the underlying C data type. Ie, a 4-bit integer can be
// stored in an int8_t, but the value range is from -8 to 7.
template <DataType storage_type, DataType expressed_type>
inline constexpr StorageType<storage_type> Quantize(
    StorageType<expressed_type> expressed_value,
    StorageType<storage_type> zero_point, StorageType<expressed_type> scale_inv,
    StorageType<storage_type> min_value = Storage<storage_type>::kMinValue,
    StorageType<storage_type> max_value = Storage<storage_type>::kMaxValue) {
  return Quantize(expressed_value, zero_point, scale_inv, min_value, max_value);
}

// Converts quantized elements which represent integer values of the storage
// type into corresponding floating-point values of the expressed type using
// the zero point and scale associated with the quantized element type.
template <typename StorageT, typename ExpressedT>
inline constexpr ExpressedT Dequantize(StorageT quantized_value,
                                       StorageT zero_point, ExpressedT scale) {
  auto sub = quantized_value - zero_point;
  return static_cast<ExpressedT>(sub) * scale;
}
}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZE_H_
