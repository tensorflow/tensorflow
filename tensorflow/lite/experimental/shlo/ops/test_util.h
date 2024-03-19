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

#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// We use a vector class that is different from std::vector to have a consistent
// API when dealing with bool tensors.
template <class T>
using Vector = absl::InlinedVector<T, 1>;

template <DataType storage_type, typename = void>
struct Distribution;

template <>
struct Distribution<DataType::kI1, void>
    : std::uniform_int_distribution<int32_t> {
  using std::uniform_int_distribution<int32_t>::uniform_int_distribution;
};

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
  absl::c_generate(vec, [&] {
    if constexpr (storage_type == DataType::kI1) {
      return dist(rd) >= 0;
    } else {
      return dist(rd);
    }
  });
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

template <DataType... Types>
struct TestParam;

template <DataType storage_type>
struct TestParam<storage_type> {
  static constexpr DataType kStorage = storage_type;
  using StorageT = StorageType<storage_type>;
};

template <DataType storage_type, DataType expressed_type>
struct TestParam<storage_type, expressed_type> {
  static constexpr DataType kStorage = storage_type;
  static constexpr DataType kExpressed = expressed_type;
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
};

// Typed test parameter tag to ask for a per-tensor quantized tensor.
template <class TestParamT>
struct PerTensor {
  using Param = TestParamT;
};

// Typed test parameter tag to ask for a per-channel quantized tensor.
template <class TestParamT, Axis kAxis = 0>
struct PerAxis {
  using Param = TestParamT;
  static constexpr Axis axis = kAxis;
};

constexpr const char* ToString(DataType t) {
  switch (t) {
    case DataType::kI1:
      return "I1";
      break;
    case DataType::kSI4:
      return "SI4";
      break;
    case DataType::kSI8:
      return "SI8";
      break;
    case DataType::kSI16:
      return "SI16";
      break;
    case DataType::kSI32:
      return "SI32";
      break;
    case DataType::kBF16:
      return "BF16";
      break;
    case DataType::kF16:
      return "F16";
      break;
    case DataType::kF32:
      return "F32";
      break;
  }
  return "Unknown data type";
}

template <class T>
struct ParamName;

template <DataType T, DataType... Ts>
struct ParamName<TestParam<T, Ts...>> {
  static std::string Get() {
    std::string name = std::string("") + ToString(T);
    ((name += std::string("_") + ToString(Ts)), ...);
    return name;
  }
};

template <DataType T, DataType... Ts>
struct ParamName<PerTensor<TestParam<T, Ts...>>> {
  static std::string Get() {
    std::string name = std::string("PerTensor[") + ToString(T);
    ((name += std::string("_") + ToString(Ts)), ...);
    return name + "]";
  }
};

template <DataType T, DataType... Ts, Axis axis>
struct ParamName<PerAxis<TestParam<T, Ts...>, axis>> {
  static std::string Get() {
    std::string name = std::string("PerAxis[") + ToString(T);
    ((name += std::string("_") + ToString(Ts)), ...);
    return name + ":" + std::to_string(axis) + "]";
  }
};

template <class TestParamT, class... TestParamTs>
struct ParamName<std::tuple<TestParamT, TestParamTs...>> {
  static std::string Get() {
    std::string name = ParamName<TestParamT>::Get();
    ((name += std::string(":") + ParamName<TestParamTs>::Get()), ...);
    return name;
  }
};

class TestParamNames {
 public:
  template <class T>
  static std::string GetName(int) {
    return ParamName<T>::Get();
  }
};

template <template <class> class F, class T>
struct Map;

template <template <class> class F, class... Ts>
struct Map<F, ::testing::Types<Ts...>> {
  using Types = ::testing::Types<F<Ts>...>;
};

template <template <class> class F, class T>
using MapTypes = typename Map<F, T>::Types;

template <class... Ts>
struct Concat;

template <class... Ts>
struct Concat<::testing::Types<Ts...>> {
  using Types = ::testing::Types<Ts...>;
};

template <class... Ts, class... Us, class... ExtraTypes>
struct Concat<::testing::Types<Ts...>, ::testing::Types<Us...>, ExtraTypes...> {
  using Types =
      typename Concat<::testing::Types<Ts..., Us...>, ExtraTypes...>::Types;
};

template <class... Ts>
using ConcatTypes = typename Concat<Ts...>::Types;

template <class Op, class T>
struct WithOp;

template <class Op, class... Ts>
struct WithOp<Op, ::testing::Types<Ts...>> {
  using Types = ::testing::Types<std::tuple<Op, Ts>...>;
};

template <class Op, class T>
using WithOpTypes = typename WithOp<Op, T>::Types;

// Use this with TYPED_TEST_SUITE for boolean testing.
using BoolTestType = ::testing::Types<TestParam<DataType::kI1>>;

// Use this with TYPED_TEST_SUITE for non quantized integer testing.
using IntTestTypes =
    ::testing::Types<TestParam<DataType::kSI4>, TestParam<DataType::kSI8>,
                     TestParam<DataType::kSI16>, TestParam<DataType::kSI32>>;

// Use this with TYPED_TEST_SUITE for non quantized floating point testing.
using FloatTestTypes =
    ::testing::Types<TestParam<DataType::kBF16>, TestParam<DataType::kF16>,
                     TestParam<DataType::kF32>>;

// Use this with TYPED_TEST_SUITE for non quantized integer testing.
using NonQuantizedBoolIntTestTypes =
    testing::Types<TestParam<DataType::kI1>, TestParam<DataType::kSI4>,
                   TestParam<DataType::kSI8>, TestParam<DataType::kSI16>,
                   TestParam<DataType::kSI32>>;

// Use this with TYPED_TEST_SUITE for non quantized integer testing.
using NonQuantizedBoolFloatTestTypes =
    testing::Types<TestParam<DataType::kI1>, TestParam<DataType::kBF16>,
                   TestParam<DataType::kF16>, TestParam<DataType::kF32>>;

// Use this with TYPED_TEST_SUITE for non quantized testing.
using ArithmeticTestTypes = ConcatTypes<IntTestTypes, FloatTestTypes>;

// Use this with TYPED_TEST_SUITE for unspecified quantized testing.
using QuantizedTestTypes =
    ::testing::Types<TestParam<DataType::kSI4, DataType::kF32>,
                     TestParam<DataType::kSI8, DataType::kF32>,
                     TestParam<DataType::kSI16, DataType::kF32>,
                     TestParam<DataType::kSI4, DataType::kBF16>,
                     TestParam<DataType::kSI8, DataType::kBF16>,
                     TestParam<DataType::kSI4, DataType::kF16>,
                     TestParam<DataType::kSI8, DataType::kF16>>;

// Use this with TYPED_TEST_SUITE for quantized per tensor testing.
using PerTensorQuantizedTestTypes = MapTypes<PerTensor, QuantizedTestTypes>;

template <class T>
using PerAxis0 = PerAxis<T, 0>;

// Use this with TYPED_TEST_SUITE for quantized per axis testing.
using PerAxisQuantizedTestTypes = MapTypes<PerAxis0, QuantizedTestTypes>;

// Builds a TensorType object and returns it in a variant that can be passed to
// a tensor.
template <DataType storage_type>
TensorTypeVariant TensorTypeFor(TestParam<storage_type>, const Shape& shape) {
  return TensorType{.shape = shape, .element_type = storage_type};
}

// Builds a per tensor QuantizedTensorType object and returns it in a variant
// that can be passed to a tensor.
//
// WARNING: the scale and zero point are randomly generated:
//   - scale is in [0.5, 1.5]
//   - zero_point is in [-5, 5]
template <DataType storage_type, DataType expressed_type>
TensorTypeVariant TensorTypeFor(
    PerTensor<TestParam<storage_type, expressed_type>>, const Shape& shape) {
  std::random_device rd;
  Distribution<expressed_type> expressed_dist(0.5, 1.5);
  Distribution<storage_type> storage_dist(-5, 5);
  StorageType<expressed_type> scale =
      static_cast<StorageType<expressed_type>>(expressed_dist(rd));
  StorageType<storage_type> zero_point = storage_dist(rd);
  return QuantizedTensorType{
      .shape = shape,
      .element_type =
          QuantizedTensorElementType::PerTensor<storage_type, expressed_type>(
              scale, zero_point)};
}

// Builds a per axis QuantizedTensorType object and returns it in a variant
// that can be passed to a tensor.
//
// WARNING: scales and zero points are unspecified and may be empty.
template <DataType storage_type, DataType expressed_type, Axis axis>
TensorTypeVariant TensorTypeFor(
    PerAxis<TestParam<storage_type, expressed_type>, axis>,
    const Shape& shape) {
  return QuantizedTensorType{
      .shape = shape,
      .element_type =
          QuantizedTensorElementType::PerAxis<storage_type, expressed_type>(
              /*scales=*/{}, /*zero_points=*/{}, axis)};
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TEST_UTIL_H_
