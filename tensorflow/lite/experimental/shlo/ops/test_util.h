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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TEST_UTIL_H_

#include <random>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"

namespace shlo_ref {

template <class T>
using Vector = absl::InlinedVector<T, 1>;

template <DataType storage_type, typename = void>
struct Distribution;

template <DataType storage_type>
struct Distribution<storage_type, std::enable_if_t<IsInteger(storage_type)>>
    : std::uniform_int_distribution<typename Storage<storage_type>::Type> {
  using std::uniform_int_distribution<
      typename Storage<storage_type>::Type>::uniform_int_distribution;
};

template <DataType storage_type>
struct Distribution<storage_type, std::enable_if_t<IsFloat(storage_type)>>
    : std::uniform_real_distribution<float> {
  using std::uniform_real_distribution<float>::uniform_real_distribution;
};

template <DataType storage_type, class Config = Storage<storage_type>>
Vector<typename Config::Type> RandomBuffer(
    const Shape& shape, const typename Config::Type min = Config::kMinValue,
    const typename Config::Type max = Config::kMaxValue) {
  Vector<typename Config::Type> vec(shape.NumElements());
  std::random_device rd;
  Distribution<storage_type> dist(min, max);
  absl::c_generate(vec, [&] { return dist(rd); });
  return vec;
}

template <DataType storage_type, class Config = Storage<storage_type>>
Vector<typename Config::Type> IotaBuffer(
    const Shape& shape, const typename Config::Type start = Config::kMinValue,
    const typename Config::Type min = Config::kMinValue,
    const typename Config::Type max = Config::kMaxValue) {
  Vector<typename Config::Type> vec(shape.NumElements());
  auto v = start;
  for (auto& e : vec) {
    e = v;
    if (++v > max) {
      v = min;
    }
  }
  return vec;
}

template <DataType storage_type, DataType expressed_type = DataType::kF32>
struct TestParam {
  static constexpr DataType kStorage = storage_type;
  static constexpr DataType kExpressed = expressed_type;
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
};

// Use this with TYPED_TEST_SUITE for non quantized testing.
using NonQuantizedTestTypes =
    testing::Types<TestParam<DataType::kSI4>, TestParam<DataType::kSI8>,
                   TestParam<DataType::kSI16>, TestParam<DataType::kSI32>,
                   TestParam<DataType::kBF16>, TestParam<DataType::kF16>,
                   TestParam<DataType::kF32>>;

// Use this with TYPED_TEST_SUITE for quantized testing.
using QuantizedTestTypes =
    testing::Types<TestParam<DataType::kSI4, DataType::kF32>,
                   TestParam<DataType::kSI8, DataType::kF32>,
                   TestParam<DataType::kSI16, DataType::kF32>,
                   TestParam<DataType::kSI4, DataType::kBF16>,
                   TestParam<DataType::kSI8, DataType::kBF16>,
                   TestParam<DataType::kSI4, DataType::kF16>,
                   TestParam<DataType::kSI8, DataType::kF16>>;

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TEST_UTIL_H_
