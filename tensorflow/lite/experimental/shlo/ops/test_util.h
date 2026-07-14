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
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// We use a vector class that is different from std::vector to have a consistent
// API when dealing with bool tensors.
template <class T>
using Vector = absl::InlinedVector<T, 1>;

// Helper for UniformDistribution.
template <DataType storage_type, typename = void>
struct UniformDistributionImpl;

template <>
struct UniformDistributionImpl<DataType::kI1, void>
    : std::uniform_int_distribution<int32_t> {
  using std::uniform_int_distribution<int32_t>::uniform_int_distribution;
};

template <>
struct UniformDistributionImpl<DataType::kSI4, void>
    : std::uniform_int_distribution<int8_t> {
  using std::uniform_int_distribution<int8_t>::uniform_int_distribution;
};

template <DataType storage_type>
struct UniformDistributionImpl<storage_type,
                               std::enable_if_t<IsInteger(storage_type)>>
    : std::uniform_int_distribution<typename Storage<storage_type>::Type> {
  using std::uniform_int_distribution<
      typename Storage<storage_type>::Type>::uniform_int_distribution;
};

template <DataType storage_type>
struct UniformDistributionImpl<storage_type,
                               std::enable_if_t<IsFloat(storage_type)>>
    : std::uniform_real_distribution<float> {
  using std::uniform_real_distribution<float>::uniform_real_distribution;
};

// Helps creating a uniform distribution for the given data type.
template <DataType storage_type>
using UniformDistribution = UniformDistributionImpl<storage_type>;

// Returns a vector filled with random data according to the set distribution.
template <DataType storage_type,
          template <DataType> class Distribution = UniformDistribution,
          class MinT = StorageType<storage_type>,
          class MaxT = StorageType<storage_type>,
          class Config = Storage<storage_type>>
Vector<typename Config::Type> RandomBuffer(const Shape& shape,
                                           const MinT min = Config::kMinValue,
                                           const MaxT max = Config::kMaxValue) {
  using StorageT = StorageType<storage_type>;
  const StorageT min_val =
      min > Config::kMinValue ? static_cast<StorageT>(min) : Config::kMinValue;
  const StorageT max_val =
      max < Config::kMaxValue ? static_cast<StorageT>(max) : Config::kMaxValue;
  Vector<typename Config::Type> vec(shape.NumElements());
  std::random_device rd;
  if constexpr (std::is_same_v<I4, StorageT>) {
    Distribution<DataType::kSI8> dist(min_val, max_val);
    absl::c_generate(vec, [&] { return static_cast<StorageT>(dist(rd)); });
  } else {
    Distribution<storage_type> dist(min_val, max_val);
    absl::c_generate(vec, [&] { return dist(rd); });
  }
  return vec;
}

// Returns a vector filled with incremental value. The values wrap around
// according to the storage type range.
template <DataType storage_type, class StartT = StorageType<storage_type>,
          class MinT = StorageType<storage_type>,
          class MaxT = StorageType<storage_type>,
          class Config = Storage<storage_type>>
Vector<typename Config::Type> IotaBuffer(const Shape& shape,
                                         const StartT start = Config::kMinValue,
                                         const MinT min = Config::kMinValue,
                                         const MaxT max = Config::kMaxValue) {
  using StorageT = StorageType<storage_type>;
  const StorageT min_val =
      min > Config::kMinValue ? static_cast<StorageT>(min) : Config::kMinValue;
  const StorageT max_val =
      max < Config::kMaxValue ? static_cast<StorageT>(max) : Config::kMaxValue;
  Vector<typename Config::Type> vec(shape.NumElements());
  StorageT v = start >= min_val ? static_cast<StorageT>(start) : min_val;
  v = v <= max_val ? v : min_val;
  for (auto& e : vec) {
    e = v;
    if (v >= max_val) {
      v = min_val;
    } else {
      ++v;
    }
  }
  return vec;
}

// Typed test parameter type.
template <DataType... Types>
struct TestParam;

// Typed test parameter specialization for non quantized tensors.
template <DataType storage_type>
struct TestParam<storage_type> {
  static constexpr DataType kStorage = storage_type;
  using StorageT = StorageType<storage_type>;
};

// Typed test parameter specialization for quantized tensors.
template <DataType storage_type, DataType expressed_type>
struct TestParam<storage_type, expressed_type> {
  static constexpr DataType kStorage = storage_type;
  static constexpr DataType kExpressed = expressed_type;
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
};

// Typed test parameter tag to ask for a per-tensor quantized tensor.
//
// TestParamT should be a `TestParam<storage_type, expressed_type>`.
template <class TestParamT>
struct PerTensor {
  using Param = TestParamT;
};

// Typed test parameter tag to ask for a per-channel quantized tensor.
//
// TestParamT should be a `TestParam<storage_type, expressed_type>`.
template <class TestParamT, Axis kAxis = 0>
struct PerAxis {
  using Param = TestParamT;
  static constexpr Axis axis = kAxis;
};

// Helps getting a human readable typed test parameter name.
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

// Allows GTest to print a human readable version of the typed test parameters.
class TestParamNames {
 public:
  template <class T>
  static std::string GetName(int) {
    return ParamName<T>::Get();
  }
};

// Applies the F template to the given testing::Types list.
template <template <class> class F, class T>
struct Map;

template <template <class> class F, class... Ts>
struct Map<F, ::testing::Types<Ts...>> {
  using Types = ::testing::Types<F<Ts>...>;
};

template <template <class> class F, class T>
using MapTypes = typename Map<F, T>::Types;

// Concatenates testing::Types lists.
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

// Transforms a list of types into a list of tuple<Op, type>.
template <class Op, class T>
struct WithOp;

template <class Op, class... Ts>
struct WithOp<Op, ::testing::Types<Ts...>> {
  using Types = ::testing::Types<std::tuple<Op, Ts>...>;
};

template <class Op, class T>
using WithOpTypes = typename WithOp<Op, T>::Types;

// Helps generating a cross-product of lists.
template <class Accu, class... Lists>
struct CrossProductImpl;

template <class... AccuTs, class... Ts, class... Lists>
struct CrossProductImpl<::testing::Types<AccuTs...>, ::testing::Types<Ts...>,
                        Lists...> {
  using Types =
      ConcatTypes<typename CrossProductImpl<::testing::Types<AccuTs..., Ts>,
                                            Lists...>::Types...>;
};

template <class... AccuTs>
struct CrossProductImpl<::testing::Types<AccuTs...>> {
  using Types = ::testing::Types<::testing::Types<AccuTs...>>;
};

// Generates a cross-product of lists.
template <class... Lists>
struct CrossProduct {
  using Types = typename CrossProductImpl<::testing::Types<>, Lists...>::Types;
};

template <class... Lists>
using CrossProductTypes = typename CrossProduct<Lists...>::Types;

static_assert(
    std::is_same_v<
        CrossProductTypes<::testing::Types<int, float>,
                          ::testing::Types<char, double>>,
        ::testing::Types<
            ::testing::Types<int, char>, ::testing::Types<int, double>,
            ::testing::Types<float, char>, ::testing::Types<float, double>>>);

static_assert(
    std::is_same_v<
        CrossProductTypes<::testing::Types<int>, ::testing::Types<char, double>,
                          ::testing::Types<float>>,
        ::testing::Types<::testing::Types<int, char, float>,
                         ::testing::Types<int, double, float>>>);

// Filters out the types that don't satisfy the predicate.
template <template <class...> class Predicate, class List>
struct Filter;

template <template <class...> class Predicate, class... Ts>
struct Filter<Predicate, ::testing::Types<Ts...>> {
  using Type =
      ConcatTypes<std::conditional_t<Predicate<Ts>::value, ::testing::Types<Ts>,
                                     ::testing::Types<>>...>;
};

template <template <class...> class Predicate, class List>
using FilterTypes = typename Filter<Predicate, List>::Type;

static_assert(std::is_same_v<
              FilterTypes<std::is_integral, ::testing::Types<int, char, float>>,
              ::testing::Types<int, char>>);

// Checks if all given types are the same.
template <class T, class... Ts>
struct SameTypes : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};

// Checks if all types in the testing::Types list are the same.
template <class T, class... Ts>
struct SameTypes<::testing::Types<T, Ts...>> : SameTypes<T, Ts...> {};

// Provides a new predicate that negates the given one.
template <template <class...> class Pred>
struct NegatePred {
  template <class... Ts>
  using Predicate = std::negation<Pred<Ts...>>;
};

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

// Customization point for generic tests that need to create a supported tensor
// for an op but that don't care what that type is.
//
// Specialize this in the test file if F32 isn't supported by the op under test.
template <class Op>
struct SupportedOpDataType {
  static constexpr DataType kStorageType = DataType::kF32;
};

// Customization point for generic tests that need to create a supported output
// tensor for an op but that don't care what that type is.
//
// Specialize this in the test file if `SupportedOpDataType<Op>::kStorageType`
// isn't supported by the op under test.
template <class Op>
struct SupportedOpOutputDataType {
  static constexpr DataType kStorageType =
      SupportedOpDataType<Op>::kStorageType;
};

// Customization point for generic tests that need a valid attribute
// configuration to create an op but that don't care what that configuration is.
//
// Specialize this in the test file if F32 isn't supported by the op under test.
template <class Op>
struct SupportedOpAttributes {
  static typename Op::Attributes Get() { return {}; };
};

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
  UniformDistribution<expressed_type> expressed_dist(0.5, 1.5);
  UniformDistribution<storage_type> storage_dist(-5, 5);
  StorageType<expressed_type> scale =
      static_cast<StorageType<expressed_type>>(expressed_dist(rd));
  StorageType<storage_type> zero_point =
      StorageType<storage_type>(storage_dist(rd));
  return QuantizedPerTensorTensorType{
      .shape = shape,
      .element_type = QuantizedElementTypePerTensor(storage_type, zero_point,
                                                    expressed_type, scale)};
}

// Builds a per axis QuantizedTensorType object and returns it in a variant
// that can be passed to a tensor.
//
// WARNING: scales and zero points are unspecified and may be empty.
template <DataType storage_type, DataType expressed_type, Axis axis>
TensorTypeVariant TensorTypeFor(
    PerAxis<TestParam<storage_type, expressed_type>, axis>,
    const Shape& shape) {
  return QuantizedPerAxisTensorType{
      .shape = shape,
      .element_type = QuantizedElementTypePerAxis(storage_type, {},
                                                  expressed_type, {}, axis)};
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TEST_UTIL_H_
