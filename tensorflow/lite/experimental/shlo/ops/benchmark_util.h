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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BENCHMARK_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BENCHMARK_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/f16.h"

namespace shlo_ref {

// Converts the given number of Kibibytes to the equivalent number of bytes.
// This is useful for specifying test input sizes as `KiB(8)`.
constexpr size_t KiB(size_t kibibytes) { return kibibytes * 1024; }

template <DataType data_type, typename T = StorageType<data_type>>
std::vector<T> GenerateRandomVector(size_t size) {
  std::vector<T> data(size);
  if constexpr (std::is_integral_v<T>) {
    static std::uniform_int_distribution<T> distribution(
        Storage<data_type>::kMinValue, Storage<data_type>::kMaxValue);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  } else if constexpr (std::is_floating_point_v<T>) {
    static std::uniform_real_distribution<T> distribution(-1.0f, 1.0f);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  } else {
    static_assert(std::is_same_v<T, BF16> || std::is_same_v<T, F16>);
    static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  }
  return data;
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BENCHMARK_UTIL_H_
