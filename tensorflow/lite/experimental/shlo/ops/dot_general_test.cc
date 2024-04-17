/*Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::FloatEq;
using testing::Pointwise;
namespace shlo_ref {

namespace {
template <class T>
struct NonQuantizedFloatDotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedFloatDotGeneralTest, FloatTestTypes,
                 TestParamNames);

TYPED_TEST(NonQuantizedFloatDotGeneralTest, FloatTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2});
  const Shape shape_rhs({2, 2, 2});
  const Shape shape_lb({1});
  const Shape shape_rb({1});
  const Shape shape_lc({1});
  const Shape shape_rc({1});
  const Shape shape_r({2, 2, 2});

  Vector<float> lhs_data_float{1.1, 2.2, 3.3, 4.3, 5.5, 6, 7, 8};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<int32_t> lhsb_dim{0};
  std::vector<int32_t> rhsb_dim{0};
  std::vector<int32_t> lhsc_dim{2};
  std::vector<int32_t> rhsc_dim{1};

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{
      .type = TensorType{.shape = shape_lb, .element_type = DataType::kSI32},
      .data = lhsb_dim.data()};
  Tensor rhs_batching_dimensions{
      .type = TensorType{.shape = shape_rb, .element_type = DataType::kSI32},
      .data = rhsb_dim.data()};
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = shape_lc, .element_type = DataType::kSI32},
      .data = lhsc_dim.data()};
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = shape_rc, .element_type = DataType::kSI32},
      .data = rhsc_dim.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});

  Vector<StorageT> expected_data;
  if (std::is_same<StorageT, float>::value) {
    Vector<float> expected_data_float = {1.32, 2.64, 3.96, 5.16,
                                         6.6,  7.2,  8.4,  9.6};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else if (std::is_same<StorageT, F16>::value) {
    Vector<float> expected_data_float = {1.319, 2.639, 3.96, 5.16,
                                         6.6,   7.203, 8.4,  9.6};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else {
    Vector<float> expected_data_float = {1.328, 2.656, 3.968, 5.18,
                                         6.625, 7.218, 8.437, 9.6};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  }

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedFloatDotGeneralTest, FloatTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2, 2});
  const Shape shape_rhs({2, 2, 2, 2});
  const Shape shape_lb({2});
  const Shape shape_rb({2});
  const Shape shape_lc({1});
  const Shape shape_rc({1});
  const Shape shape_r({2, 2, 2, 2});

  Vector<float> lhs_data_float{1.1,  2.2,   3.3,   4.3,   5.5,   6,   7,   8,
                               11.1, 12.22, 33.33, 44.32, 15.15, 6.6, 7.3, 8.1};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2,
                               1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<int32_t> lhsb_dim{0, 3};
  std::vector<int32_t> rhsb_dim{0, 3};
  std::vector<int32_t> lhsc_dim{2};
  std::vector<int32_t> rhsc_dim{2};

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{
      .type = TensorType{.shape = shape_lb, .element_type = DataType::kSI32},
      .data = lhsb_dim.data()};
  Tensor rhs_batching_dimensions{
      .type = TensorType{.shape = shape_rb, .element_type = DataType::kSI32},
      .data = rhsb_dim.data()};
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = shape_lc, .element_type = DataType::kSI32},
      .data = lhsc_dim.data()};
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = shape_rc, .element_type = DataType::kSI32},
      .data = rhsc_dim.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});

  Vector<StorageT> expected_data;
  if (std::is_same<StorageT, float>::value) {
    Vector<float> expected_data_float = {
        1.32,  1.32,  6.6,   6.6,   5.16,   5.16,   9.6,  9.6,
        13.32, 13.32, 18.18, 18.18, 53.184, 53.184, 9.72, 9.72};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else if (std::is_same<StorageT, F16>::value) {
    Vector<float> expected_data_float = {
        1.319, 1.319, 6.6,   6.6,   5.16,   5.16,   9.6,    9.6,
        13.32, 13.32, 18.18, 18.18, 53.184, 53.184, 9.7265, 9.7265};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else {
    Vector<float> expected_data_float = {
        1.328,  1.328,  6.625, 6.625, 5.187, 5.187, 9.6,  9.6,
        13.375, 13.375, 18.25, 18.25, 53.25, 53.25, 9.75, 9.75};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  }

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedFloatDotGeneralTest, FloatTestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shape_lhs({4, 4});
  const Shape shape_rhs({4});
  const Shape shape_lc({1});
  const Shape shape_rc({1});
  const Shape shape_r({4});
  Vector<float> lhs_data_float{
      2.9270215,    7.86154318,   -5.63383484, 1.18890381,
      1.66500914,   -0.686581432, -1.0598495,  3.66114569,
      -2.12638235,  -5.93207598,  1.81490195,  0.333228439,
      -0.129492328, 5.85269737,   1.17887712,  -3.05277419};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{0.148809016, 4.21798277, -8.70141696,
                               -2.01860809};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<int32_t> lhsc_dim{1};
  std::vector<int32_t> rhsc_dim{0};
  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{};
  Tensor rhs_batching_dimensions{};
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = shape_lc, .element_type = DataType::kSI32},
      .data = lhsc_dim.data()};
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = shape_rc, .element_type = DataType::kSI32},
      .data = rhsc_dim.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});
  Vector<StorageT> expected_data;
  if (std::is_same<StorageT, float>::value) {
    Vector<float> expected_data_float = {80.2178345, -0.816446066, -41.8026962,
                                         20.5717602};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else if (std::is_same<StorageT, F16>::value) {
    Vector<float> expected_data_float = {80.250000, -8.242180e-01,
                                         -4.181250e+01, 2.057810e+01};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else {
    Vector<float> expected_data_float = {8.000000e+01, -7.812500e-01,
                                         -4.175000e+01, 2.050000e+01};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  }
  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntDotGeneralTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2});
  const Shape shape_rhs({2, 2});
  const Shape shape_r({2, 2, 2, 2});
  Vector<int32_t> lhs_data_int{1, 2, 3, 4};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int32_t> rhs_data_int{1, 0, 0, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{};
  Tensor rhs_batching_dimensions{};
  Tensor lhs_contracting_dimensions{};
  Tensor rhs_contracting_dimensions{};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});

  Vector<int32_t> expected_data_int{1, 0, 0, 1, 2, 0, 0, 2,
                                    3, 0, 0, 3, 4, 0, 0, 4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shape_lhs({7, 3, 4});
  const Shape shape_rhs({7, 4});
  const Shape shape_lb({1});
  const Shape shape_rb({1});
  const Shape shape_lc({1});
  const Shape shape_rc({1});
  const Shape shape_r({7, 3});
  Vector<int32_t> lhs_data_int{
      0,  1,  4,  1,  -2, -3, 0, 0, 6,  -1, 0,  0,  1,  0,  -2, 0,  1,
      3,  4,  -6, 2,  4,  4,  0, 0, -2, -1, 1,  -2, -3, 0,  2,  -3, 0,
      0,  -2, 4,  -7, 2,  2,  0, 4, 2,  0,  -6, 1,  1,  2,  -2, -2, 0,
      -1, -4, -1, 0,  -1, 1,  3, 1, 1,  -4, 0,  0,  1,  -1, 0,  4,  -2,
      0,  5,  0,  -1, 0,  2,  1, 2, -1, 1,  -3, -2, -6, -3, -1, -3};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int32_t> rhs_data_int{2,  0, -1, 4,  -4, 0,  2,  -1, 0, 6,
                               8,  0, -1, -3, -1, -1, -3, 0,  5, 0,
                               -3, 0, 3,  -1, 2,  1,  -2, -3};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<int32_t> lhsb_dim{0};
  std::vector<int32_t> rhsb_dim{0};
  std::vector<int32_t> lhsc_dim{2};
  std::vector<int32_t> rhsc_dim{1};
  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{
      .type = TensorType{.shape = shape_lb, .element_type = DataType::kSI32},
      .data = lhsb_dim.data()};
  Tensor rhs_batching_dimensions{
      .type = TensorType{.shape = shape_rb, .element_type = DataType::kSI32},
      .data = rhsb_dim.data()};
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = shape_lc, .element_type = DataType::kSI32},
      .data = lhsc_dim.data()};
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = shape_rc, .element_type = DataType::kSI32},
      .data = rhsc_dim.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});
  Vector<int32_t> expected_data_int{0,   -4, 12, -8,  10, 0,  -20,
                                    -18, 0,  13, -14, 0,  6,  12,
                                    2,   11, 17, 1,   -6, 11, -4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());
  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4, 2});
  const Shape shape_rhs({4, 2});
  const Shape shape_lc({1});
  const Shape shape_rc({1});
  const Shape shape_r({4, 4});
  Vector<int32_t> lhs_data_int{1, 2, 3, 4, 5, 6, 7, 8};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int32_t> rhs_data_int{2, 1, 1, 2, 2, 2, 1, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<int32_t> lhsc_dim{1};
  std::vector<int32_t> rhsc_dim{1};

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor lhs_batching_dimensions{};
  Tensor rhs_batching_dimensions{};
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = shape_lc, .element_type = DataType::kSI32},
      .data = lhsc_dim.data()};
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = shape_rc, .element_type = DataType::kSI32},
      .data = rhsc_dim.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions});

  Vector<int32_t> expected_data_int{4,  5,  6,  3,  10, 11, 14, 7,
                                    16, 17, 22, 11, 22, 23, 30, 15};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
