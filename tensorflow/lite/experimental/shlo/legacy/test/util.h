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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_TEST_UTIL_H_

#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace testing {

template <ElementType storage_type, ElementType expressed_type>
std::vector<typename Storage<storage_type>::Type> QuantizeVector(
    const std::vector<typename Storage<expressed_type>::Type>& input,
    const QuantizedParameter& quantized_parameter) {
  std::vector<typename Storage<storage_type>::Type> result;
  typename Storage<expressed_type>::Type scale_inv =
      1.0 / quantized_parameter.scale;
  for (auto x : input) {
    auto q = QuantizePartial<storage_type, expressed_type>(
        x, scale_inv, quantized_parameter.zero_point);
    result.push_back(q);
  }
  CHECK_OK(CompleteQuantization<storage_type>(  // Crash OK
      result.data(), result.size(),
      /* storage_min */ std::nullopt,
      /* storage_min */ std::nullopt));
  return result;
}

}  // namespace testing

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_TEST_UTIL_H_
