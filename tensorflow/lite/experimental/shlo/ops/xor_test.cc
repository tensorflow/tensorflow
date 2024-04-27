/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions xor
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/xor.h"

#include <functional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {

template <>
struct ParamName<XorOp> {
  static std::string Get() { return "Xor"; }
};

template <DataType>
struct Xor : std::bit_xor<void> {};

template <>
struct Xor<DataType::kI1> {
  template <class T>
  bool operator()(T lhs, T rhs) const {
    return static_cast<bool>(lhs) != static_cast<bool>(rhs);
  }
};

template <>
struct SupportedOpDataType<XorOp> {
  static constexpr DataType kStorageType = DataType::kSI32;
};

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(Xor, BinaryElementwiseOpShapePropagationTest,
                               XorOp, TestParamNames);

using MultipyBaselineContraintTypes = BinaryElementwiseBaselineConstraintTypes<
    XorOp, ConcatTypes<BoolTestType, BaselineConstraintIntTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    Xor, BinaryElementwiseSameBaselineElementTypeConstraintTest,
    MultipyBaselineContraintTypes, TestParamNames);

using UnsupportedTypes =
    WithOpTypes<XorOp, ConcatTypes<FloatTestTypes, PerTensorQuantizedTestTypes,
                                   PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Xor, BinaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

using SupportedTypes = ConcatTypes<BoolTestType, IntTestTypes>;

template <class T>
struct XorTest : ::testing::Test {};

TYPED_TEST_SUITE(XorTest, SupportedTypes, TestParamNames);

TYPED_TEST(XorTest, ArithmeticTestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/1, /*max=*/5);
  Vector<StorageT> output_data(shape.NumElements());
  Tensor lhs_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = lhs_data.data()};
  Tensor rhs_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(lhs_data, rhs_data, expected_data.begin(),
                    Xor<TypeParam::kStorage>());

  auto op = Create(XorOp::Attributes{});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
