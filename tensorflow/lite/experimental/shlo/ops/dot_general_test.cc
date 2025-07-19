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

using testing::Eq;
using testing::FloatEq;
using testing::FloatNear;
using testing::Pointwise;
namespace shlo_ref {

namespace {

template <class T>
struct NonQuantizedBoolDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedBoolDotGeneralTest, BoolTestType, TestParamNames);

TYPED_TEST(NonQuantizedBoolDotGeneralTest, BoolTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({7, 3, 4});
  const Shape shape_rhs({7, 4});
  const Shape shape_r({7, 3});

  Vector<StorageT> lhs_data = {
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true};
  Vector<StorageT> rhs_data = {true, true, true, true, true, true, true,
                               true, true, true, true, true, true, true,
                               true, true, true, true, true, true, true,
                               true, true, true, true, true, true, true};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {true, true, true, true, true, true, true,
                                    true, true, true, true, true, true, true,
                                    true, true, true, true, true, true, true};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedBoolDotGeneralTest, BoolTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({3, 4});
  const Shape shape_rhs({4, 2});
  const Shape shape_r({3, 2});

  Vector<StorageT> lhs_data = {true, true,  true,  false, true, true,
                               true, false, false, true,  true, true};
  Vector<StorageT> rhs_data = {true, true,  true, false,
                               true, false, true, false};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {true, true, true, true, true, false};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

template <class T>
struct NonQuantizedIntDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntDotGeneralTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2});
  const Shape shape_rhs({2, 2});
  const Shape shape_r({2, 2, 2, 2});
  Vector<int64_t> lhs_data_int{1, 2, 3, 4};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{1, 0, 0, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{};
  Vector<Axis> rhsc_dim{};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{1, 0, 0, 1, 2, 0, 0, 2,
                                    3, 0, 0, 3, 4, 0, 0, 4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shape_lhs({7, 3, 4});
  const Shape shape_rhs({7, 4});
  const Shape shape_r({7, 3});

  Vector<int64_t> lhs_data_int{
      0,  1,  4,  1,  -2, -3, 0, 0, 6,  -1, 0,  0,  1,  0,  -2, 0,  1,
      3,  4,  -6, 2,  4,  4,  0, 0, -2, -1, 1,  -2, -3, 0,  2,  -3, 0,
      0,  -2, 4,  -7, 2,  2,  0, 4, 2,  0,  -6, 1,  1,  2,  -2, -2, 0,
      -1, -4, -1, 0,  -1, 1,  3, 1, 1,  -4, 0,  0,  1,  -1, 0,  4,  -2,
      0,  5,  0,  -1, 0,  2,  1, 2, -1, 1,  -3, -2, -6, -3, -1, -3};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{2,  0, -1, 4,  -4, 0,  2,  -1, 0, 6,
                               8,  0, -1, -3, -1, -1, -3, 0,  5, 0,
                               -3, 0, 3,  -1, 2,  1,  -2, -3};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{0,   -4, 12, -8,  10, 0,  -20,
                                    -18, 0,  13, -14, 0,  6,  12,
                                    2,   11, 17, 1,   -6, 11, -4};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4, 2});
  const Shape shape_rhs({4, 2});
  const Shape shape_r({4, 4});

  Vector<int64_t> lhs_data_int{1, 2, 3, 4, 5, 6, 7, 8};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{2, 1, 1, 2, 2, 2, 1, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<int64_t> expected_data_int{4,  5,  6,  3,  10, 11, 14, 7,
                                    16, 17, 22, 11, 22, 23, 30, 15};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntDotGeneralTest, IntTestTypesTensorsWork4) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 3, 4});
  const Shape shape_rhs({1, 4, 3});
  const Shape shape_r({1});

  Vector<int64_t> lhs_data_int{2, 0, 0, 0, 5, -3, 0, 4, -1, 0, 0, -1};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{0, 4, 2, 3, 3, 3, -6, -2, 1, -1, 1, 0};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2, 1};
  Vector<Axis> rhsc_dim{1, 2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{13};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;
template <class T>
struct NonQuantizedkF32DotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF32DotGeneralTest, kF32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4});
  const Shape shape_rhs({4});
  const Shape shape_r({1});

  Vector<StorageT> lhs_data{-1.73818827, 6.32115507, 2.81545162, -1.37914991};
  Vector<StorageT> rhs_data{-4.02553225, -2.70646834, 3.14252234, 1.59961236};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{0};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {-3.46935892};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatNear(0.00001), expected_data));
}

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({8, 4, 3, 3, 4});
  const Shape shape_rhs({4, 8, 3, 4, 2});
  const Shape shape_r({8, 4, 3, 2});

  Vector<StorageT> lhs_data{
      -1.87649846,   -3.47338486,   -2.48692966,   -7.18852519,   0.453702927,
      5.35152435,    2.1862061,     1.54857051,    0.730003654,   0.950800061,
      0.0558503307,  0.0404239409,  2.07866955,    -1.83830249,   4.63644028,
      -1.980930e-01, 1.74757624,    0.240779877,   -2.8026557,    2.06706452,
      -0.814424097,  1.22022116,    -1.61731887,   0.413890779,   7.942410e-01,
      5.37683344,    2.1054821,     2.8406539,     -7.01480246,   0.500502944,
      0.120232038,   -1.68313301,   -2.57794762,   -3.83692598,   1.70738566,
      1.73775625,    1.89522338,    0.780784369,   -2.97426748,   -2.35742474,
      0.649545252,   4.74978542,    4.22566509,    -4.88821459,   -6.85589647,
      -6.1821456,    0.0498935468,  -2.40693545,   -3.70844388,   6.42948675,
      -0.643269539,  -3.15726566,   1.8571384,     1.83920121,    3.94780493,
      -1.64931226,   -0.592977166,  0.0820118338,  -8.18413925,   4.22887039,
      5.79892874,    -3.17882061,   -6.14308691,   4.43086195,    -4.28587151,
      -3.73970294,   2.15639758,    -3.24867964,   -0.100068301,  -0.352196336,
      -0.0349913202, 1.43447292,    3.139117,      -0.914029896,  2.3963964,
      2.25374484,    -1.96811771,   4.82946491,    2.02067089,    3.65482736,
      -2.69943142,   -3.375340e+00, 0.348427474,   -2.39984655,   1.82775664,
      -0.202018514,  -4.63333035,   -0.453364611,  2.79687309,    -0.385181457,
      0.0693540424,  2.57960486,    0.0463522561,  1.67748439,    0.67565459,
      -0.32969445,   1.10405314,    -2.659693,     -1.95852602,   0.724824428,
      5.12927294,    0.127019018,   -4.68311453,   -1.62116969,   -4.4265914,
      1.92160928,    -0.261960953,  -3.44796705,   -1.84308767,   2.20060563,
      -1.12443507,   -3.90431213,   -1.13467884,   -2.80576205,   1.12600482,
      6.01857567,    0.483005375,   -1.42877698,   2.54888725,    1.01370716,
      -0.368566573,  1.90416145,    -2.70484948,   -0.733804941,  -2.51124692,
      -1.53261483,   -1.84946156,   -1.91693854,   -0.875394821,  1.50137925,
      3.99593663,    -3.53555632,   1.924613,      2.48561096,    -6.58027649,
      4.23332453,    0.487618297,   -2.74514627,   4.94205236,    -4.15321207,
      0.174054608,   -1.53711438,   -4.00133467,   0.788624882,   -2.5709312,
      -2.68854904,   1.43690276,    -2.4834075,    1.11090064,    -6.32847214,
      -1.46099293,   -7.23574305,   -1.74951851,   3.99956036,    -0.530959547,
      4.58668375,    -5.11389971,   2.94757533,    -4.33469582,   -5.96422195,
      -0.443479359,  -1.22048128,   -3.0721488,    0.477036893,   -1.20768142,
      -1.36174417,   -3.04397416,   -0.867913842,  0.143732369,   3.92143393,
      1.60633636,    2.60106683,    -1.04948568,   1.59614825,    1.91751957,
      0.895943403,   1.1568327,     -1.50673008,   1.70746088,    -0.371906698,
      2.515030e+00,  5.72573185,    2.16121387,    -0.36456579,   -0.883925437,
      -0.210071921,  -0.730625689,  0.734956443,   -3.58134866,   1.01085246,
      0.832847177,   -1.17692459,   -0.572514236,  1.28458679,    -1.38431895,
      0.0370868035,  -0.314202487,  1.61585808,    -2.62398028,   -4.89429235,
      6.49681568,    -2.01633072,   7.63070106,    2.31477237,    -0.946075439,
      1.40081429,    -0.870581269,  4.71911716,    1.2298311,     -0.494098604,
      -4.08609343,   -2.20851278,   -4.98212528,   6.5466876,     5.3053441,
      -1.23897469,   1.5525645,     3.13576913,    3.52707529,    0.723028421,
      5.702620e-01,  -6.23285818,   -0.536408663,  -2.10483956,   -1.96350431,
      0.508728564,   -3.708640e+00, 1.87505436,    -5.34865665,   3.20490074,
      7.3808279,     2.83542085,    -0.12711592,   -2.43619037,   2.03934121,
      3.2360909,     -4.02856302,   -1.97660112,   -6.05021572,   -0.691315532,
      -2.40139151,   3.43819237,    -3.66166639,   -4.05007219,   -2.41310406,
      -1.67347085,   1.20363843,    -2.05918622,   -2.72777462,   -0.80741769,
      0.493756652,   4.51686621,    -0.9707762,    2.68249464,    -3.09376621,
      1.4653995,     -4.56097507,   -4.58768129,   -4.77981091,   1.12059927,
      -4.90206957,   2.60315108,    0.936685681,   -1.64452827,   -0.334013283,
      -3.0882659,    -4.05892372,   1.10093117,    2.55691791,    5.77829885,
      -1.33279729,   -0.916881918,  2.6500392,     -0.788509905,  -0.527036846,
      3.61783266,    2.43902516,    2.35868311,    1.00640929,    1.20368254,
      -3.39374757,   0.666641294,   1.16126561,    1.325180e+00,  0.194145083,
      2.63131952,    2.56542611,    1.27865839,    2.81528592,    2.68246031,
      0.222708434,   -1.39180505,   -6.6826272,    -4.26872206,   6.4082365,
      0.761401057,   -9.20273113,   1.38191938,    0.840463995,   4.77528858,
      -1.68857718,   1.35805297,    1.25696886,    -1.21014082,   -1.66435266,
      1.04348385,    0.0584274493,  0.932799637,   -1.66938519,   -5.21581268,
      5.24336481,    3.01468277,    0.521278918,   -3.28188252,   4.84264278,
      -2.35329652,   1.0029794,     -7.51446342,   2.22791839,    -1.16119504,
      -3.571710e-01, 3.27321649,    -1.59462273,   7.83165836,    0.973350644,
      1.59425402,    -0.676703989,  1.82872117,    1.51141095,    0.0414509065,
      2.0381484,     -7.84299231,   -6.93860149,   -0.281864792,  3.16317153,
      4.10116196,    -1.66568196,   2.64648438,    3.03480029,    2.87710953,
      -2.55378103,   3.90833902,    2.31282496,    -1.13470304,   -3.33487153,
      -0.11133723,   -4.30292749,   -3.38045907,   -1.47063899,   -7.04785156,
      -4.49502325,   -1.20632243,   2.96336031,    2.75589705,    0.186776072,
      0.166558504,   0.710282922,   2.80123496,    -5.45793152,   -4.1064415,
      -0.881354451,  -4.42291355,   5.94807482,    -0.845859885,  -5.42477083,
      5.73047876,    2.22986841,    -1.68050218,   -3.74641609,   0.846037149,
      -3.9808619,    -2.71278071,   1.47646236,    -2.80977511,   -0.939890623,
      -2.6122539,    -2.61204338,   -1.62924218,   4.06698227,    -2.51858044,
      -3.36010194,   -2.97220564,   -1.49709046,   -0.485943943,  -0.948599159,
      3.85410213,    1.02384865,    0.501251221,   -1.37325025,   -0.290856361,
      0.538159728,   0.221178874,   -3.07363248,   0.396877468,   6.51339674,
      -2.10960937,   0.0190884694,  -1.09555709,   -2.51101828,   -2.31213427,
      -0.339224517,  0.587753296,   1.71227682,    -4.37930393,   3.37113452,
      -0.857002735,  1.66821659,    1.87198615,    4.02758408,    2.58188057,
      -2.02887321,   -2.43621469,   -0.426523447,  -3.47489071,   -0.753577053,
      -1.00958252,   -4.83628464,   2.99669075,    5.93932152,    -0.139120638,
      1.642960e+00,  -1.15994322,   -3.0145998,    0.664293408,   -0.271663308,
      -1.76385379,   2.22600842,    0.951898813,   5.79541349,    2.07155633,
      -2.81737041,   -3.1079545,    4.12475634,    0.656094193,   -3.569695,
      6.72006369,    -1.10828245,   -0.747320353,  -0.164212227,  0.553490043,
      5.29554844,    -0.127745867,  -1.49527371,   7.8846879,     -2.79885125,
      4.14194918,    4.76525307,    0.772530257,   -3.41649508,   1.37163103,
      1.38833356,    -1.25138903,   -0.210216954,  -3.69565415,   -2.70914745,
      2.28796172,    0.322550386,   -1.57452941,   -3.41677809,   -1.91440475,
      -3.96095228,   2.8715806,     -1.56754804,   -3.13352394,   2.07228661,
      1.69402659,    -0.506832361,  -2.03662825,   -3.30853581,   2.68716598,
      -0.372595102,  0.890035629,   0.698648572,   1.87839651,    0.505521119,
      3.10566497,    0.821987509,   -0.500042498,  -3.30381083,   4.37517166,
      -3.57771492,   1.44336319,    -1.52336943,   0.37912032,    5.4609642,
      3.46656346,    2.25183821,    1.27542913,    4.66861963,    -3.41533923,
      -3.53427124,   -4.12347031,   1.39569986,    -0.822389066,  0.807919741,
      -1.06021965,   2.69650221,    2.65054083,    -0.762958884,  -1.88367844,
      1.41521394,    -0.347926617,  -2.15650892,   6.931410e-01,  3.17373705,
      -4.56752396,   1.05384219,    0.642139554,   2.86379194,    -0.655686438,
      5.87414074,    -3.19704819,   -2.97449756,   2.23151112,    -0.52391094,
      6.94013548,    -1.26480377,   0.921559095,   -0.791909158,  -2.15799189,
      5.130990e-01,  -0.82301712,   -2.9171319,    -3.80701542,   0.488462418,
      0.726314783,   -4.98142481,   4.78956461,    2.4482398,     -0.88653171,
      -3.04181528,   -3.31807733,   -2.8385129,    -2.12279606,   -0.877318561,
      2.56024504,    -2.53635812,   -4.44041348,   -0.825987875,  0.951859652,
      2.24447775,    0.842470645,   4.59782696,    -2.92269897,   -2.44534302,
      5.80306435,    -1.81401789,   3.695260e+00,  -0.961050927,  -2.54540062,
      2.19440055,    3.16854882,    1.60922539,    1.10178757,    1.20156217,
      -2.8990128,    2.99109268,    0.0469676852,  -2.82498407,   -2.12242913,
      -0.578763306,  -0.632457315,  -1.22698808,   -4.48224831,   4.55396843,
      -4.24374342,   2.93350387,    -1.27676558,   -0.781452536,  1.0025599,
      -0.759562432,  -1.78930914,   5.66538429,    2.44931078,    -1.85138834,
      -1.74100351,   1.04861712,    -1.87689626,   3.29311919,    -2.3766675,
      -3.17583919,   1.35815358,    2.7775526,     -3.54416275,   -3.78612757,
      4.99057961,    4.27979231,    -9.926170e-01, 2.36855245,    0.345492154,
      -0.189878732,  4.1841712,     2.36518168,    -0.583524406,  -5.179190e+00,
      -2.09180689,   0.0538935512,  -2.41012168,   0.295729131,   2.40852165,
      5.14292479,    2.64033461,    1.29260194,    0.741406381,   -3.82073593,
      -1.37954152,   0.966273367,   2.16947842,    -0.56433475,   -1.89522183,
      2.714110e-03,  2.7556076,     2.43188787,    5.3428669,     5.67797089,
      1.83574533,    -1.72462511,   2.79798269,    -1.61519849,   2.26435542,
      0.054490529,   9.76374435,    -1.74227154,   -0.747497379,  5.44391108,
      -6.27359629,   0.678475677,   4.76186371,    5.61840105,    3.54570484,
      -5.07487392,   -0.0125423213, 0.395109773,   -5.05701923,   3.0488193,
      -1.62002122,   0.0731608123,  0.552313685,   -4.22485113,   -1.19008803,
      -0.168055132,  3.37870312,    -5.2994833,    -5.74993896,   -1.1148237,
      -3.97859216,   -0.322120965,  -8.621190e-01, -0.0992122888, -1.65759921,
      1.96980274,    -2.29468799,   0.71057868,    -3.37486267,   2.53592205,
      0.856289207,   -0.722598135,  -3.43178368,   0.360708207,   -1.07858288,
      0.122655191,   -4.04899645,   -2.39308286,   -2.74635386,   0.171571046,
      -4.1185379,    5.70451975,    4.06273603,    -1.48877537,   -6.05069494,
      3.39123893,    0.524862409,   -2.88184953,   -4.77822065,   1.2852813,
      -4.1259079,    0.529391527,   3.82064414,    -0.616423368,  3.32143879,
      -4.424350e+00, 7.92192459,    -0.201763839,  1.37330735,    -2.06987882,
      -4.44536448,   -6.62122202,   -0.975232243,  -1.20360732,   2.82912183,
      -0.453562289,  2.79727888,    7.73274374,    2.2929585,     -4.70550776,
      -0.426447362,  -0.473913103,  -0.527026176,  -1.75051737,   -9.2841072,
      -2.65398598,   3.60620332,    -1.00961053,   -8.504520e-01, 5.94038486,
      -4.28642464,   -2.20273852,   -2.9992907,    -3.07135677,   0.856342494,
      5.38253689,    1.32636261,    -3.67015696,   -0.582955301,  2.4455533,
      -0.525749326,  2.16236329,    -0.680979549,  0.772522866,   1.30917096,
      0.956611573,   -1.16447878,   0.35012573,    -0.575177133,  -5.959250e-01,
      -0.670398175,  -4.28085661,   6.25828695,    -2.09919953,   -2.00302482,
      0.493205756,   1.89417124,    0.0135864187,  2.72723722,    -1.09558415,
      -1.40152466,   4.81239557,    -2.01474977,   -1.36288643,   0.775672912,
      1.55070245,    3.238100e+00,  0.739745318,   0.343425304,   -4.42305183,
      -0.904785513,  2.50790596,    -4.03653717,   -0.806768059,  -2.49821162,
      -0.470299184,  -2.60650444,   0.322934538,   0.894313574,   0.147280633,
      5.59399891,    2.34958506,    3.71105814,    1.84778965,    3.73111963,
      -4.44270802,   0.556840122,   -6.10548401,   -6.42483664,   -4.85367918,
      -2.35003185,   -7.74674845,   2.47499466,    -1.23163867,   4.32820559,
      0.389856517,   -4.299191,     -0.616833091,  -2.18287039,   -1.97331214,
      -1.58703709,   -2.00213027,   -0.110841833,  -0.0560073107, 2.184160e+00,
      -1.08201194,   -2.89953518,   -2.05458331,   3.95964289,    2.89324927,
      3.60389829,    -0.412305087,  0.232630566,   -4.90435457,   2.96322131,
      -0.612402081,  -1.30004179,   1.47677326,    0.384562671,   -3.19109344,
      -1.20662665,   -1.17860675,   -1.42060709,   -2.52035904,   1.82026672,
      -1.13306355,   -1.41466868,   -0.268506974,  -1.68247485,   3.03915024,
      0.90274769,    1.34598911,    -2.35818624,   -2.31933475,   -0.16551739,
      -2.52889752,   1.65803778,    -1.24321365,   -1.64859533,   -0.515696943,
      2.27630901,    0.85365498,    -2.16984177,   1.62544227,    -0.87252283,
      -6.06706047,   -0.880420267,  -1.19637835,   0.405364484,   -1.8160032,
      -2.01907969,   -5.78519439,   -3.27134275,   3.63473034,    3.605300e+00,
      -2.13048434,   -1.53998291,   3.56228971,    -2.96158385,   1.48613691,
      0.762356817,   -1.33678532,   -2.88666916,   0.602801799,   4.9925828,
      -0.196285829,  -0.839323937,  1.94820654,    -1.05501282,   -2.74938226,
      2.18676519,    4.25956726,    -1.08438385,   1.65356338,    -1.68874979,
      -0.545932651,  0.877064228,   0.639560043,   -1.62403548,   3.49764299,
      -4.41002607,   2.01518202,    1.1364032,     8.11913681,    3.45640326,
      -4.42684364,   -0.212173939,  1.70066559,    -4.47084904,   -0.50339365,
      2.44187331,    -4.94309139,   -0.516171813,  0.518492758,   -4.67917585,
      -4.26952457,   -3.35842943,   3.89500475,    2.28719902,    -6.73151827,
      -1.46810639,   -2.6148932,    3.1746068,     -0.0255467296, 3.89041662,
      1.68199575,    5.5268693,     -2.81873941,   0.678116738,   -0.826269745,
      2.53404927,    4.34016705,    -1.56743252,   -1.58239484,   1.87683117,
      0.846875488,   -5.60215759,   -0.327029765,  1.2753861,     4.7352047,
      -1.71447754,   -0.174289346,  -2.84831023,   -1.3919127,    -3.845930e+00,
      -4.29346037,   -2.45225954,   0.209896758,   1.7295711,     3.95666456,
      2.81802821,    -2.97079134,   -4.26618528,   0.209537193,   -4.231440e+00,
      -2.69605398,   -2.1790092,    2.99482226,    6.644230e-01,  2.10070586,
      -0.887188971,  4.55098152,    -2.51289797,   -0.21171017,   -0.267657042,
      1.16942096,    -3.25197268,   -6.08889771,   -0.967430055,  3.63926458,
      -0.573276639,  -2.88880324,   -5.68361282,   1.84854674,    1.3873359,
      7.13644695,    -3.35623503,   0.475173175,   -0.642963111,  -5.68893814,
      -0.930827617,  2.98797631,    1.98726499,    2.62143159,    2.03542018,
      -0.681307256,  -4.22476912,   0.524580538,   0.118409932,   0.519303262,
      2.06678462,    1.38492334,    0.416094035,   0.148540586,   -4.25225687,
      -2.35143661,   -1.1586498,    1.3036505,     -2.46451426,   1.22584414,
      -1.28577876,   -2.96878862,   -2.93611741,   4.35444498,    -0.436907023,
      -2.84679461,   -0.409445763,  -2.1352005,    7.29348755,    0.85128957,
      -0.530047238,  -0.538847089,  -2.19303846,   -3.90233278,   3.21010137,
      -1.9948436,    3.1756916,     -1.79694438,   0.7727198,     -4.6075511,
      3.79669261,    -2.14621425,   0.111959659,   1.65004325,    1.472180e+00,
      2.60391235,    2.4350481,     -6.48943949,   -5.64670372,   -1.98698616,
      -4.04967308,   1.16454268,    1.41787112,    -1.06886733,   -3.40769744,
      6.21486759,    -0.535509169,  -0.427331626,  -1.5331018,    4.2751503,
      -1.52771103,   -1.31707144,   2.50824976,    -3.89739275,   1.11267865,
      2.30178928,    4.71923208,    -4.05298328,   -4.04065371,   0.065517731,
      -4.1748414,    -0.0681283771, 0.885989785,   0.0353716053,  -3.97524118,
      -0.417335689,  -3.18925619,   1.32857823,    -0.926950097,  -0.731239378,
      3.22948623,    -0.0955532342, -4.02142668,   5.86452103,    -0.707449972,
      0.689531266,   -2.48676848,   -6.44853926,   1.64957929,    -0.0197120663,
      -1.39228702,   4.07486534,    3.247180e-01,  -0.866697311,  1.74094832,
      3.94280124,    -1.1883601,    1.14525628,    1.50722468,    1.69494152,
      0.0182054918,  2.07163858,    1.29938054,    -6.36976957,   2.75823927,
      2.63221741,    4.09661865,    -1.37980103,   1.22187674,    -1.44791365,
      -1.69895697,   5.57939148,    0.541170418,   -2.06188869,   -0.661397576,
      -2.5885191,    0.0399168245,  -0.941184103,  -1.24745119,   6.36060667,
      1.31126094,    3.796760e-01,  -5.52618265,   -2.79564595,   -4.58214378,
      3.84463167,    -0.866135597,  -5.79306555,   -2.00669932,   2.09216857,
      -1.0426383,    -2.44722366,   -1.65651631,   0.491471529,   -1.72548485,
      -0.207864463,  -4.07416773,   1.53459513,    -2.503760e+00, 4.84056377,
      1.26055753,    -0.143832132,  -0.607564092,  -1.84999263,   -1.96739912,
      3.84209466,    1.05214381,    -1.04142284,   -0.457963943,  -0.077251263,
      -0.929952263,  -3.62113237,   2.49142241,    0.704675137,   2.9799037,
      -2.02257752,   0.485854566,   -0.985855758,  2.41958928,    -0.989334821,
      2.09791303,    -8.46319484,   -0.750644385,  2.72333717,    -2.98349071,
      -1.855165,     0.0713950693,  2.46981239,    -4.362610e+00, 6.17007875,
      -3.392694,     -0.251924962,  -0.39285475,   -1.00826025,   -3.52503896,
      -2.96945834,   1.27461243,    0.636823297,   2.39904618,    3.66743135,
      1.57293701,    1.94727623,    -1.59285128,   -2.75286055,   2.82671714,
      1.27283013,    -1.50429761,   -1.96161699,   -0.244579434,  -3.79650617,
      -8.856260e+00, -3.39907169,   1.26923954,    3.57903409,    1.01408863,
      -4.02438354,   -0.792661249,  0.568224728,   1.83581483,    2.91377783,
      1.05539167,    -2.71168709,   -1.2345444,    1.5107305,     -0.0355182588,
      -1.89490092,   2.22541857,    0.00441914238, -0.789098501,  1.31387424,
      3.83794498,    2.65786266,    -7.53286743,   -0.636833608,  0.00183040428,
      0.722873687,   1.71402884};
  Vector<StorageT> rhs_data{
      -1.24657238,   1.01384413,     -3.68404102,   -0.65727967,
      -2.20306873,   -3.60879111,    -3.8555665,    -3.41769433,
      2.32689977,    -0.651230156,   2.3286674,     -0.170348749,
      -3.06489849,   -2.31551123,    1.3469137,     -0.850648522,
      -4.14321709,   -5.14827967,    2.17822051,    1.74889326,
      -1.39003062,   -1.49137175,    2.88233376,    2.48932648,
      -3.22695971,   -2.555750e+00,  -7.97081851,   1.55227101,
      -2.01674628,   6.291230e-01,   4.55689812,    1.25129855,
      0.4463512,     3.09087038,     4.40772867,    0.974942743,
      -3.93725038,   2.133310e+00,   0.352029532,   -1.60659218,
      -3.64822745,   -1.13424337,    -3.37435341,   0.117033109,
      5.82059669,    -0.00486629643, -1.59028792,   -0.977053046,
      -3.49896789,   -3.45762157,    -0.450610042,  0.637396574,
      -0.0107554533, 1.559680e+00,   1.15201771,    -1.82857966,
      1.30176389,    6.3158946,      2.0208056,     -0.271234959,
      -0.947180509,  3.93136716,     2.84945703,    -3.06478739,
      -3.38732624,   4.24206305,     -4.49562454,   -0.720611155,
      1.89444768,    7.17448378,     -2.5224905,    0.226725951,
      -2.18071842,   2.91119528,     0.608982741,   6.45959473,
      -0.834486067,  -0.219553247,   1.47552228,    -2.38614178,
      -3.58561492,   -1.9354856,     1.29494917,    2.56405711,
      0.777223408,   -1.39358866,    -6.14588118,   3.03597498,
      -2.24201965,   3.14030933,     3.74432421,    2.62377763,
      -0.755063116,  1.12070894,     -1.42078614,   2.61160207,
      -6.348104,     -1.45728636,    0.754401743,   2.93093657,
      -0.160488173,  3.03028536,     -1.5415349,    -2.97050118,
      -5.36658716,   -1.5714314,     -0.865237951,  6.01657534,
      5.532690e+00,  2.9789238,      0.931426525,   -5.85052443,
      1.07972062,    0.92937386,     -1.75694561,   0.801552414,
      -2.49874687,   1.76685226,     -0.951398193,  2.68063641,
      2.43348622,    0.936484456,    3.98135161,    2.49458289,
      4.49959087,    -2.16274905,    -0.0303466395, 2.91059756,
      2.5509336,     0.669066727,    9.15212345,    -1.70583487,
      -0.682069421,  3.26616931,     4.66344547,    1.33111107,
      5.09694481,    -2.85288358,    -0.0176183078, 1.96957016,
      0.408689052,   0.772002696,    1.00494063,    3.4464283,
      3.3235476,     2.586980e+00,   -4.82196331,   -0.952813327,
      0.845038056,   2.78020501,     3.2594409,     4.43915319,
      -4.73968077,   3.594650e+00,   -1.25507247,   -5.03880692,
      0.994898557,   -3.24184346,    -0.445916593,  -0.622709751,
      3.18088937,    -5.47800255,    -3.73844147,   -0.646125436,
      5.79001617,    0.500494957,    2.23914528,    -3.48805237,
      -1.72740126,   0.896240651,    2.84088349,    -2.20048237,
      -0.85975033,   6.0348835,      5.35066414,    -2.11828876,
      -3.49717855,   -7.69965076,    -0.367580116,  0.804826915,
      3.58174086,    -5.79524279,    -1.51893628,   0.824159562,
      9.20310306,    5.31374025,     6.34586954,    2.28396773,
      -0.403258353,  5.12537098,     4.3031292,     -2.65290284,
      -5.2636776,    -3.51487565,    -1.42686224,   1.88877463,
      -0.262296468,  4.65288448,     7.14556932,    2.58636856,
      2.34597397,    3.71459889,     -3.58157873,   2.83496189,
      -2.3332715,    -0.338360935,   -2.69046712,   -2.33796573,
      -4.4157238,    -0.500537574,   2.38809037,    -2.14437985,
      -3.11489964,   0.140777841,    -6.3196125,    1.34418774,
      -2.1494348,    -1.18559301,    -1.94842494,   -3.965040e+00,
      0.675582826,   5.72968054,     2.83464122,    1.62541378,
      -2.68843818,   2.72441339,     3.48913789,    -3.32298183,
      -2.98955607,   -1.55624235,    -1.75817704,   -0.857959628,
      0.471723408,   -3.54938817,    -1.9963733,    -0.924630641,
      -7.83587694,   3.71666241,     -1.41741896,   0.335636705,
      5.06878042,    -3.03472233,    2.49280381,    -0.747685611,
      -3.074054,     5.69408655,     0.626221716,   -1.73778772,
      0.863461554,   -4.68878508,    -1.78167379,   -0.639935433,
      3.06881142,    -0.248560607,   -2.12070155,   0.319562286,
      -2.04849768,   -5.0141387,     5.67170382,    -3.0827992,
      1.31209314,    1.50726068,     -6.537600e-01, -0.440410912,
      -0.53784585,   -2.65779376,    3.287462,      3.53356409,
      -2.55797958,   -2.11108088,    -7.167370e+00, 0.342814505,
      -2.20540547,   -0.0577974692,  1.27131796,    4.50894117,
      -0.114468418,  -0.693277537,   2.06324053,    -5.14415741,
      -0.445552289,  0.436530411,    -2.46582389,   -2.54455781,
      -0.246206298,  -0.728103458,   0.0946381688,  -0.693409144,
      -2.81080079,   0.376707077,    -1.63657367,   -0.848980247,
      -3.33079958,   -2.128304,      4.80996132,    -1.52053046,
      0.561844885,   -4.074471,      3.99621224,    1.20584619,
      0.0355783701,  1.77770412,     2.5489316,     -0.771665513,
      3.61207199,    0.513344765,    2.25575471,    1.55538023,
      3.19059944,    3.36188793,     -3.88276744,   2.20113325,
      -2.12703085,   0.186662823,    -2.11997437,   2.81382585,
      2.82580829,    -5.50706768,    -1.50315809,   0.115999289,
      6.3384943,     -2.16788745,    -0.112780444,  4.50515413,
      3.59243965,    -1.67986524,    3.52740026,    6.10385323,
      1.42135417,    -2.05201602,    -1.99551427,   7.59026622,
      -5.62916756,   1.90418124,     -4.73673964,   -0.361628562,
      -0.57901454,   3.13404632,     1.56160569,    3.56668353,
      -1.36695623,   0.312280864,    -1.28146577,   4.016650e+00,
      3.29637694,    -1.43006957,    4.51899672,    4.7697897,
      0.20095259,    2.38167787,     1.99440634,    -1.00506067,
      -1.00603402,   3.7956872,      -5.17561913,   -0.0865147784,
      4.43065357,    0.418808758,    1.18090987,    1.88717675,
      -4.33498621,   -4.16117096,    0.798016667,   -2.29897022,
      0.00295607653, -1.48870671,    -0.312802702,  1.36883199,
      -3.17801619,   -2.25399947,    0.34082678,    -1.4882232,
      2.68805838,    2.35586596,     1.33206594,    0.675621033,
      -6.7105875,    4.10348272,     3.28692842,    -5.15189409,
      -0.0948765054, 6.90435123,     3.05468297,    5.43353939,
      -0.385962278,  -0.0692735463,  0.318759739,   -1.00344741,
      0.216391906,   -3.72218943,    8.68399238,    -1.1224463,
      4.30681181,    2.5260942,      2.21805286,    -0.898484408,
      -0.434577137,  -3.53358412,    2.11606193,    -1.74562836,
      -1.54590452,   -0.109068751,   0.153739899,   -1.38152695,
      -0.209432751,  1.14506042,     4.66124296,    -1.35665607,
      0.0105530201,  5.77236748,     2.68208623,    -8.29340171,
      -3.229420e-01, 1.56024289,     -0.504419088,  1.69938648,
      1.22519374,    1.25800431,     -2.65492892,   2.93285775,
      -0.422579587,  -0.583059788,   -0.43677479,   4.344270e+00,
      2.61894727,    3.00653982,     2.25923038,    -5.07813597,
      -1.43019366,   1.19754398,     -0.023538392,  -2.28989816,
      -0.193483606,  1.21312201,     -0.675268888,  -2.688260e+00,
      0.301961511,   6.06459618,     -3.27444935,   -0.144141719,
      3.76491427,    -2.08499265,    0.566690743,   2.70845079,
      1.20209599,    0.855298519,    6.39038372,    0.346128374,
      0.35429734,    -0.67405951,    -3.54524016,   1.11381853,
      0.779215812,   -3.91729069,    -1.93394482,   2.9198575,
      3.92481208,    0.832618952,    -6.59708548,   0.395623118,
      2.83876491,    1.73538589,     -4.13551092,   1.00660586,
      -1.55244744,   -5.491290e+00,  2.22253227,    4.33824682,
      -1.1372329,    -1.10546863,    -7.223880e-01, -1.13420784,
      -3.43628359,   1.62771928,     -6.65144348,   3.399225,
      -6.1184988,    -0.662647306,   1.610919,      -1.48636508,
      -1.18255925,   -1.54418683,    2.47549939,    3.30986834,
      -5.09933138,   -8.96811389,    4.08811426,    -2.95877385,
      -3.56650519,   1.84365368,     -0.0231283419, 1.12171495,
      1.38181889,    2.45867538,     -0.146814212,  3.82596731,
      -2.19869852,   4.64434099,     0.932928502,   -0.479400814,
      0.744050443,   1.45107627,     4.55596209,    6.746943,
      2.42777467,    0.0941791087,   2.13297677,    0.113510385,
      3.4913137,     1.00779891,     0.0751199722,  6.48089075,
      -0.553477526,  2.54101563,     -1.58880281,   4.57484579,
      -3.62128758,   -4.83006048,    -7.46897792,   -0.810882687,
      3.10893726,    -4.13703442,    -4.3874588,    0.378020555,
      3.28033137,    -3.02286386,    -1.54140484,   -1.46719217,
      0.25948897,    4.55920744,     -1.64648533,   4.63023806,
      5.79556274,    -3.7605021,     3.37880945,    -12.7813787,
      3.88836575,    4.68949556,     0.21831584,    4.084170e-01,
      2.83094859,    0.6698969,      3.77472615,    3.01003933,
      -1.83598173,   -1.25939715,    -3.82819605,   0.232888922,
      1.92648757,    -0.323625535,   -4.26971531,   7.22254372,
      1.29273283,    -4.19372416,    0.511550128,   5.64667082,
      -0.567642212,  2.10657692,     -2.24055433,   0.253761411,
      -7.025200e-01, 0.360233426,    2.921450e+00,  -4.38879824,
      3.64558101,    0.25900051,     7.28601027,    -3.57347083,
      -1.09795022,   -0.0355491862,  1.35900795,    -1.57337916,
      2.67705154,    -1.17924929,    -2.88141966,   -1.11120248,
      -0.174536526,  1.31345987,     -3.88873887,   0.364970833,
      -6.25203848,   -3.58217835,    -1.05437648,   2.28839612,
      0.624963581,   5.6699791,      3.70031619,    2.05081058,
      -1.87853837,   2.31541371,     0.844914495,   -4.5047245,
      1.42396808,    0.779891669,    5.586530e+00,  -1.87148917,
      3.76229048,    0.104935072,    -0.982089221,  0.407636106,
      1.97547483,    -1.87161291,    -4.69943619,   -0.92255795,
      2.35603714,    2.99566436,     -1.57622588,   2.86368155,
      -1.79465377,   0.804181218,    -4.35109901,   0.352856249,
      0.572360456,   -2.40418816,    -2.24349785,   0.88425821,
      4.41774607,    -0.543589234,   2.2110858,     0.356733561,
      2.38534737,    -1.21506286,    -3.99008203,   2.25092959,
      -5.06516647,   -2.76328039,    -9.23355197,   0.414836168,
      -3.37221885,   -0.723610818,   2.87493062,    2.118040e+00,
      5.53781414,    -1.28999054,    2.27397084,    -1.85480726,
      2.01036429,    1.49733067,     -3.19647026,   -1.3390125,
      -2.50632644,   2.23404455,     1.65315735,    5.2423234,
      1.00297499,    -1.8663013,     -4.54012299,   -1.26409924,
      1.07230961,    -2.81381106,    1.08460224,    0.31691879,
      -2.54406476,   -2.95141315,    0.669420778,   1.35821414,
      -4.43546629,   -3.83563972,    -2.02853036,   -1.23030865,
      1.43963671,    1.9356581,      -0.197113857,  6.5183382,
      -4.95763397,   -5.90738726,    -0.246024892,  4.75100517,
      -2.85628796,   1.01210165,     -0.605603814,  -1.15972781,
      -0.969795584,  2.05170798,     -1.15825438,   1.42321622,
      -7.1069026,    -1.11835086,    -4.39312315,   2.68677855,
      -3.99955344,   6.22028827,     2.31590486,    -0.4073838,
      -1.53607488,   1.72166872,     0.735597789,   -0.656856536,
      -3.50190902,   1.853770e+00,   -2.66098094,   1.3175329,
      -1.14394188,   -6.39037132,    -0.0406936146, -1.99089503,
      3.37045383,    -1.69182765,    -2.62958813,   -1.73084259,
      0.333444238,   1.25549245,     5.03755379,    -0.142416865,
      3.55430198,    1.01736009,     1.20021832,    1.50660932,
      -9.827470e-01, -0.130734086,   0.238607258,   2.14576173,
      -4.74845171,   -0.973471403,   2.07720232,    3.35224771,
      4.984930e+00,  1.4296577,      3.22820497,    -6.41438532,
      -2.39855337,   1.664230e+00,   1.26695669,    -1.99027085,
      -2.45068216,   -0.402405024,   -0.554093361,  0.465462655,
      -1.47549593,   0.011701745,    1.18621385,    2.65923762,
      3.69845843,    1.63287652,     0.446511567,   -0.472108245,
      -3.33471131,   1.20922971,     2.27674651,    3.65219188,
      0.254563302,   -2.40063739,    -1.95164299,   -3.68525386,
      -1.87393534,   0.336540788,    0.327130884,   3.90053129,
      0.614907205,   -5.15305567,    0.223802373,   2.9089098,
      2.26630831,    -0.0829703286,  -2.80919266,   -2.91985059,
      5.77795315,    3.84300613,     0.0136255184,  -1.27516901,
      3.7247982,     1.82866955,     -1.18727612,   -0.131878763,
      2.83965039,    0.092409797,    -1.11850524,   0.138695642};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0, 1};
  Vector<Axis> rhsb_dim{1, 0};
  Vector<Axis> lhsc_dim{4, 3};
  Vector<Axis> rhsc_dim{3, 2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {
      56.318306,   24.2588539,    20.205019,    5.821990e-01, -49.2233772,
      -6.05811405, -8.78516387,   14.2142506,   -17.4677753,  33.3056946,
      4.97737217,  -60.4377708,   20.0734253,   -24.9685745,  9.41819667,
      19.3926582,  16.6954269,    43.4956436,   -3.9106915,   -39.4855804,
      30.6051826,  -9.32236289,   -20.1392651,  66.8953399,   -26.1794491,
      3.93930483,  -26.0789719,   -0.352433503, -12.3813047,  9.8152523,
      -22.1652508, -0.620383382,  -35.1543846,  -2.28662014,  -32.07444,
      42.4285088,  26.582489,     -53.9280357,  2.4155612,    -43.292038,
      4.51829052,  -86.4158096,   -8.32997894,  0.220027447,  -31.6104469,
      8.01850891,  2.78245401,    -3.7990036,   -19.3840981,  -56.2431831,
      37.8652802,  32.4322243,    -58.6883697,  15.1135826,   50.3360596,
      17.318449,   -6.27355576,   35.9602051,   -4.13606453,  -48.0326195,
      -19.6942577, 86.313095,     -2.96168518,  11.0611763,   -0.366628647,
      -31.9230919, 12.0464697,    1.74660659,   -29.2902012,  -12.5347729,
      10.5653687,  37.2415771,    -18.648571,   38.3432846,   14.0140476,
      12.6956425,  39.2983665,    8.85236358,   14.4786739,   12.4860191,
      9.40736293,  41.3088036,    -3.47433043,  15.9223089,   42.414772,
      -25.3519478, 3.075940e+01,  -3.63443398,  64.2192154,   -22.577034,
      -17.3944855, -8.64235687,   -9.84368515,  -38.1255951,  -29.6926765,
      -57.6063309, 0.391165435,   57.5635567,   -62.7238083,  -20.7641716,
      -2.84953618, -4.803700e+00, 9.8296175,    -5.00178719,  38.760994,
      -29.9148788, -4.88074398,   0.393952131,  -28.7478752,  -17.6381302,
      0.119878054, 45.7106743,    -11.9326324,  -9.60338115,  44.8364601,
      31.2282314,  67.4339141,    -18.4541168,  -39.605011,   -30.0066795,
      15.1747532,  -0.394477367,  12.8820009,   12.0523434,   -32.4030762,
      -4.62262154, -56.3963165,   -39.2059517,  -27.9977303,  -3.19525814,
      10.6836777,  47.3491745,    -3.98864579,  -2.21679211,  10.3144302,
      -6.57927989, -1.60688925,   10.8602448,   21.6038761,   -53.7793579,
      -25.9617519, -11.6823101,   1.37364864,   51.5729675,   -57.0660439,
      0.647707462, -19.8367958,   4.85115767,   -33.0055046,  -34.7681122,
      19.9960556,  -14.2741909,   -11.8272867,  -9.53694343,  -8.12217617,
      7.69343424,  9.95977783,    -14.8772993,  19.5898609,   5.1881547,
      -16.931736,  -74.0084915,   30.7655582,   49.7950897,   18.2811317,
      -4.47796917, -10.7906141,   -15.5013084,  -71.6981735,  31.1821289,
      17.1652527,  -37.9045906,   22.2522354,   45.2124405,   9.80810737,
      55.5313835,  -57.6841927,   -25.3064041,  -0.13922596,  -11.746151,
      5.87265396,  -21.13731,     9.08362102,   29.7465935,   3.98221684,
      -45.749752,  -24.1554623,   7.99809837,   -0.351560473, 23.3751907,
      9.31683254,  13.3687668};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4, 4, 3, 3, 4});
  const Shape shape_rhs({4, 4, 3, 4, 2});
  const Shape shape_r({4, 4, 3, 3, 2});

  Vector<StorageT> lhs_data{
      -5.04178476,   0.151476204,   -1.92671061,   -2.87375879,
      -1.64964902,   2.52091408,    1.24007452,    -0.762933671,
      2.12838316,    -0.673644423,  -0.703288555,  -2.59565854,
      0.33901003,    4.75170374,    0.529750645,   6.82830524,
      0.815180063,   -3.00405264,   0.708959341,   0.376890361,
      -0.296997815,  -5.73250818,   4.92057753,    5.35512066,
      -0.947735965,  -6.03644419,   -5.63391209,   -2.755400e+00,
      0.433817476,   -2.60197735,   -0.205505431,  -0.637663126,
      -0.796301245,  -1.31965911,   -0.0417316779, 0.385280967,
      -5.06596375,   8.81371689,    2.08480763,    0.598065257,
      -3.17218661,   2.39093852,    -3.13154387,   -1.31055915,
      2.37939787,    5.65154696,    -2.02114606,   -4.8075819,
      1.00599122,    0.0268017426,  -1.37520087,   -1.98449039,
      0.730953156,   -1.01289833,   2.56460834,    7.59823656,
      -4.89123106,   0.986944317,   1.29783881,    -0.171200752,
      -1.67470527,   1.51432049,    -1.01012945,   5.02613544,
      -1.3962822,    -4.51087761,   -1.47666717,   -1.645220e+00,
      1.76892245,    5.60205936,    1.15838969,    -1.9010967,
      -0.966700255,  -2.36273551,   1.21145833,    2.63929057,
      3.98323965,    1.44114912,    -0.842081964,  -4.02853537,
      -1.86561012,   -3.96083117,   -0.667696297,  1.1600076,
      -1.31863427,   1.21469092,    2.02171564,    2.70230675,
      -1.609370e+00, 3.498310e-01,  1.34420609,    4.11904192,
      2.60222268,    -2.43285298,   -6.07626152,   5.12348509,
      4.07783794,    1.20221257,    -6.04520655,   2.07061481,
      -1.95217431,   -1.78468573,   2.63070345,    -1.91144371,
      -0.911912977,  -0.186152056,  -2.49894357,   -5.76176929,
      -3.08122516,   -0.20985657,   -0.204916731,  5.166749,
      4.53348827,    -0.869878411,  -3.62227082,   2.44710708,
      0.869193851,   3.61391044,    1.1012193,     4.05431747,
      3.86989141,    2.30131507,    1.12889504,    2.44574785,
      -4.78659201,   2.7883327,     -0.741169631,  -4.94411564,
      2.91576719,    1.96097195,    -4.32673407,   2.28064942,
      -2.2770021,    -3.26711559,   0.701846242,   0.316278726,
      -1.23909247,   -0.943375885,  -4.23068619,   -2.0933435,
      4.22601795,    2.3459444,     3.1935966,     -1.14847338,
      0.588031471,   -2.85248518,   -3.3325696,    2.25423694,
      1.84813583,    0.25134176,    -2.36365581,   0.730738461,
      1.60757542,    -3.36629105,   2.46360135,    0.681054235,
      -1.71595883,   -0.0635089278, 0.945253193,   -1.00568843,
      0.819159388,   -0.300416589,  1.82366896,    -2.19864058,
      0.926027715,   0.500816941,   2.49535728,    -0.930866062,
      -0.917201757,  -2.0200069,    -2.72349143,   1.25606596,
      0.252042085,   -0.657521188,  0.560413063,   6.2234106,
      -1.37418735,   3.06129265,    -0.368260175,  -0.200462595,
      -2.3985548,    -1.37922585,   -1.30282974,   -2.8425405,
      4.74973536,    2.80821347,    -3.46784902,   0.00842234399,
      -0.247631013,  4.37321901,    -4.28235531,   1.5077858,
      0.811073958,   0.0485076681,  1.4113704,     1.3209759,
      -0.385003328,  6.61425591,    3.52686524,    0.675758183,
      -3.91932893,   0.696083724,   0.206181288,   3.42211676,
      -2.21848965,   0.00765378494, -1.73250484,   0.139556289,
      4.91744041,    -2.58870125,   -1.21156788,   3.24998927,
      -6.12808418,   0.531210899,   1.31607902,    -1.81790495,
      0.606384754,   -2.08149743,   4.00895929,    3.3647449,
      3.57985806,    4.08703566,    2.43995237,    4.43932915,
      0.0409130044,  -3.93739533,   -0.338729531,  -1.22051489,
      -4.19155359,   -2.89175248,   4.69982862,    8.86440753,
      0.88670355,    -1.81053567,   2.27680874,    -0.22210668,
      -4.9759531,    -4.7927413,    -7.65606451,   -5.30581045,
      2.62779832,    0.686693072,   1.40899336,    0.866683125,
      0.883325099,   -3.98436785,   2.75000668,    -0.396242678,
      1.48655868,    -2.83467793,   1.69456804,    -0.706806838,
      2.95579171,    3.28365779,    -2.78228974,   -2.05339599,
      -2.89153385,   0.0336176977,  -0.157525063,  -1.78963017,
      0.134164989,   -1.65353787,   1.08684969,    2.080302,
      -0.773628056,  1.91228378,    -3.75195098,   5.58550453,
      2.95131326,    3.0141778,     -5.99176883,   2.30042553,
      -4.396210e+00, -1.66954017,   -2.56444645,   1.54306984,
      1.19112384,    -0.232770041,  2.92193508,    5.51592445,
      3.87240839,    0.00348980539, 5.44295883,    -4.75972843,
      2.24310899,    0.11213059,    -2.3869338,    -4.18959093,
      0.0972665324,  -1.826244,     -2.45667601,   -3.71658635,
      -4.45040751,   4.76671171,    2.35191607,    3.54606867,
      -3.38895178,   -3.3857348,    -2.94168139,   -0.67645061,
      0.112918451,   0.920616149,   -2.78947926,   1.87949157,
      0.268920809,   -1.35844481,   1.42818141,    1.44527304,
      1.22763085,    -2.32554388,   0.473992229,   1.85849142,
      -1.4033711,    1.47591567,    -6.3808031,    -1.62420833,
      -2.92555737,   -2.23156381,   1.24626148,    -1.5713315,
      1.27316773,    0.987514078,   0.0852024927,  -0.919721842,
      -3.75747085,   0.0979869961,  -0.194479153,  1.09335494,
      -1.41294813,   1.1388073,     -1.92111301,   2.56864071,
      2.95319223,    0.73423934,    -5.57447433,   5.12217903,
      0.00995799247, -0.380946606,  -1.75406134,   -3.21676636,
      2.60667205,    1.07436407,    -7.57584476,   1.12638223,
      5.57155943,    3.03748035,    2.35524464,    -2.45726609,
      1.29659712,    -2.68127561,   -3.26663923,   1.12367952,
      -3.69241691,   2.64893651,    9.22864055,    3.68369579,
      6.412210e-02,  3.39243722,    3.47904277,    -2.54301929,
      -1.36102068,   1.28191578,    4.90938091,    -0.275454313,
      4.68056107,    -1.36586595,   -3.78845763,   -1.18090749,
      -0.415946811,  -0.631388783,  0.324598044,   2.20987511,
      5.67443228,    -3.04615927,   -4.94821548,   3.80574417,
      6.84180784,    -2.63273025,   -2.40715075,   1.93075418,
      -3.33625889,   -3.18321443,   0.541295528,   -4.76917934,
      -2.90369987,   2.31241536,    0.329516232,   4.86076403,
      3.24049449,    3.17291236,    3.2192142,     5.64272261,
      -1.32433665,   -1.82022369,   -0.230574131,  -0.978485405,
      4.44776869,    4.2145071,     -1.57052159,   -2.44703078,
      0.774934769,   2.49540138,    2.16847873,    0.836883544,
      -5.94422817,   1.106740e+00,  4.43202734,    4.00687933,
      4.014160e+00,  2.61587381,    5.61802673,    -0.051751297,
      -1.68498886,   2.96948123,    4.09686184,    -2.71713303E-4,
      0.893459141,   4.52929688,    1.34990978,    -1.49190283,
      0.394296914,   5.24419546,    -3.42225552,   4.46671152,
      -0.664941788,  5.26397562,    -2.2276175,    2.52759719,
      -4.26367903,   -2.17731977,   -3.08579373,   2.06552291,
      -1.83389866,   2.92510557,    1.60644186,    -3.70957828,
      -0.129375011,  3.42437577,    0.543603063,   -0.28785497,
      0.478800803,   -0.98436594,   0.66818732,    -1.59905159,
      2.63034654,    -2.21303439,   0.688476205,   1.02240288,
      -7.90429115,   -0.0573968291, 0.657117963,   -0.897135496,
      1.95495868,    0.490171731,   -2.93047071,   -0.11387711,
      -0.132202908,  3.52029014,    3.08778071,    1.31491721,
      -3.16870213,   -0.989759147,  4.63253975,    -2.71787286,
      0.388657898,   -2.95024633,   5.07563734,    -1.23531175,
      -0.216410354,  1.8761611,     1.86321795,    4.7839222,
      7.70282507,    0.455425292,   -1.28358495,   -3.13217378,
      -3.36567497,   -1.25449228,   0.16010493,    1.37790179,
      -4.4069314,    2.31904936,    1.86863422,    3.64978862,
      -4.34574223,   -1.54497015,   1.02404749,    -3.24121833,
      -9.53796863,   0.904418885,   -2.93947172,   3.95798016,
      -5.70081091,   3.01781058,    -2.07795572,   5.48630524,
      0.921848654,   -4.64908552,   2.622118,      8.27204227,
      -1.9748528,    0.0605070703,  -1.60740161,   -0.821350157,
      -2.0483849,    3.54479122,    -2.97693968,   -0.453617215,
      -3.92811418,   2.96640015,    -0.323537499,  3.32659984,
      -2.48478293,   2.69845319,    3.93529654,    0.622713744,
      -1.09294987,   -2.62539983,   0.769982397,   -2.12569618,
      -7.06108427,   0.822229444,   -6.849760e-02, -1.54295397,
      2.1145997,     1.5401752,     5.28202391,    1.44856083,
      -2.61290359,   0.0278399028,  1.09066629,    1.0829159,
      4.11216545,    -3.25935316,   2.04704571,    0.37205416,
      -3.41621542,   -10.7348309,   -1.7963239,    -1.86615884,
      4.93598032,    1.01911712,    2.7788012,     3.30800271,
      1.12032688,    -3.52348971,   -1.70670605,   -0.32787478,
      1.39297938,    -1.72488141,   0.327451974,   -0.0470529757,
      4.72512436,    4.43413353,    2.4570291,     -0.5909127,
      -4.235310e+00, 5.28222847,    0.773418068,   -2.34447241,
      2.10488129,    5.3777194,     0.682628452,   3.77398896,
      3.87954831,    -1.47334528,   -2.47782516,   3.3233664,
      -2.69942784,   -2.87149811,   1.69805217,    -2.12082124,
      1.541417,      -2.72026277,   1.77883863,    -0.0629182458};
  Vector<StorageT> rhs_data{
      -3.37465143,   -1.69417632,   -2.63676953,  3.48959613,    3.76929569,
      -1.70214045,   -0.534194469,  -5.6645093,   0.80516088,    -4.84139299,
      0.297968715,   2.4876616,     2.03182602,   2.95733118,    -1.09163666,
      -3.17107821,   -0.0853810086, -1.41262817,  0.370284885,   -4.08684111,
      6.25043535,    1.66273272,    0.539257944,  1.71379149,    4.64237356,
      -2.67904782,   2.97281313,    -1.44529891,  -2.84462667,   4.24389648,
      -1.02176309,   -1.69716144,   4.79662514,   2.3476634,     0.18988426,
      2.72915363,    2.42686939,    0.875523626,  1.22163475,    1.39816499,
      -1.30813563,   0.0275540408,  3.37361479,   -3.76242614,   -0.567262352,
      -3.67635751,   4.20530033,    -4.71308517,  -0.821286082,  1.2333014,
      4.1963973,     -5.29646206,   -2.0354104,   3.93841887,    -1.38728428,
      0.470515251,   -1.75695288,   -0.428026915, 0.983793914,   0.143730327,
      -3.50503016,   -6.12270117,   2.26893353,   -6.0226531,    1.97018409,
      -0.24654375,   -5.63254356,   -1.47006297,  -2.03042793,   -7.70165252,
      1.50914645,    -0.551550865,  1.54587626,   -1.06593537,   0.364829153,
      -1.96702063,   0.0572484396,  4.61271095,   3.00548053,    5.6093688,
      -3.66817141,   -3.12002683,   -0.834663867, -3.21083665,   0.400404811,
      0.687709928,   4.15773487,    1.99310601,   -1.24479079,   -3.52229595,
      3.98198652,    -4.85509968,   1.42338705,   -4.05765486,   3.48558736,
      3.21577358,    -0.60825771,   2.55980158,   -0.867806494,  5.15257359,
      -10.5400915,   3.16713905,    3.24956965,   -0.711798131,  -0.467318714,
      -0.128610581,  4.94907141,    3.94176149,   1.57009816,    6.31189346,
      0.234075353,   -1.75674891,   1.10934687,   -5.52949667,   0.380457848,
      1.46683908,    -0.567747414,  -2.67380285,  -0.402673244,  -1.66054428,
      2.92685461,    -2.00720382,   -0.772785127, -1.44571197,   5.4123292,
      2.14039302,    4.3875432,     1.10484576,   2.84443569,    1.21548033,
      -3.72020245,   1.72057557,    -0.164468676, -2.56780291,   2.58745241,
      3.22563767,    -2.66920567,   3.12807226,   5.49974966,    1.2240802,
      -4.80097771,   -2.28121734,   -2.17873335,  -0.582763553,  0.942178905,
      1.18782747,    -8.3672285,    4.06311846,   -2.48362303,   -0.511053741,
      -4.11691475,   2.58865023,    -0.50102365,  -3.19978356,   5.11805916,
      8.313450e+00,  -2.54982805,   4.87274837,   8.76603889,    -6.150220e-01,
      -1.49432862,   1.9327929,     -1.28529859,  1.56334269,    3.12380648,
      3.72169113,    1.95787656,    -1.48836613,  -0.174741551,  -2.822580e+00,
      -0.178010538,  -1.23091149,   0.715331256,  3.22940636,    -4.60552502,
      0.825909972,   -1.04398847,   4.2331562,    1.05334282,    -1.53035581,
      -0.959903359,  0.0564641543,  -2.99215627,  -3.70792842,   0.00727847638,
      -3.1699357,    0.15003638,    2.44830585,   2.88838267,    -1.99393833,
      0.523122132,   -3.58326054,   1.08154714,   4.0517664,     -4.03838873,
      -2.02566791,   2.20928812,    0.485275298,  -1.779570e+00, 0.772204339,
      1.7952075,     0.76674819,    1.38693523,   -1.02495325,   2.82639432,
      3.54748654,    -0.164700374,  -0.333710283, -1.31120992,   3.5376153,
      0.897910833,   -3.30390596,   -1.73901272,  3.8108592,     2.56733298,
      3.77585363,    -0.106362313,  2.1084938,    0.394705296,   -3.66115403,
      -5.35238409,   -3.39029837,   -3.0274291,   -4.70910311,   -4.76656961,
      1.81277156,    -1.18822575,   1.80522549,   1.4373349,     -6.79364443,
      3.79632115,    1.76385677,    -2.13693333,  -4.09047127,   -0.279991031,
      -2.34352851,   3.01021862,    -6.39942217,  3.28219414,    -0.246023327,
      0.945857942,   -0.6208781,    0.104355991,  0.257149518,   0.353006601,
      -1.05261993,   -0.136048734,  0.178120196,  3.80781913,    1.51026642,
      1.18392754,    4.15970612,    3.40225315,   -4.14681864,   -4.39334965,
      5.272150e-01,  6.16252708,    1.32139695,   -2.71472073,   1.45354366,
      -1.50192559,   4.22016954,    -1.89036655,  -0.0489338636, -1.6654737,
      0.464905411,   0.609002709,   -4.08894968,  -2.06344151,   1.18172276,
      7.32344866,    1.22370887,    -2.26924157,  -1.51449752,   1.28918338,
      -0.992235302,  -3.34665751,   -6.50869083,  2.77163506,    -0.637472391,
      -4.10131359,   -2.44303727,   -1.39476573,  -1.1677798,    -0.771763086,
      5.26347876,    1.35855722,    -2.7723062,   -0.68806833,   -1.12405348,
      -0.0598534383, 0.657204866,   0.653120637,  1.72824895,    0.172244444,
      3.38373828,    -2.17875123,   2.18062711,   -1.48428142,   1.65290797,
      5.79987574,    -1.42404222,   4.96899652,   0.506676555,   1.77456176,
      -0.515343308,  3.96922827,    -4.75212479,  -0.882124423,  0.338211805,
      -2.46625829,   0.711948573,   -0.419259965, -2.50001979,   -3.44735122,
      -1.16228914,   -8.08782196,   -0.688208758, -1.94362307,   5.94125223,
      2.43249869,    0.900148868,   -5.77259779,  1.66801929,    4.71632624,
      -1.51754785,   0.974936068,   0.777762115,  0.210504159,   -0.608417212,
      -4.42930412,   1.80843771,    3.878740e+00, 3.14484096,    3.01286364,
      0.78121227,    4.02190113,    0.0850777402, 3.27939987,    -1.6795336,
      5.1488843,     -1.48375237,   -3.19353986,  -2.08252716,   2.25357771,
      -1.93757522,   -3.22466397,   -2.96690512,  -2.53966308,   -1.28286624,
      1.13764048,    4.13580275,    -0.816939055, 0.642711222,   4.33177567,
      6.67139626,    0.648844719,   -8.19357204,  -0.753541112,  1.53278899,
      -1.89811456,   -0.693213284,  8.03378963,   0.790956139,   -3.38255167,
      1.44672048,    1.29046845,    -2.43455243,  5.3825407,     -3.70093822,
      -1.40165007,   -1.31541455,   -4.76631975,  -2.36551785,   1.6867379,
      0.864559352,   -1.84499454,   -3.90309215,  0.50124079,    1.65204442,
      3.63313937,    -3.55344439,   2.04770899,   -1.08008444};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0, 1, 2};
  Vector<Axis> rhsb_dim{0, 1, 2};
  Vector<Axis> lhsc_dim{4};
  Vector<Axis> rhsc_dim{3};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {
      10.8876619,    28.6282291,    4.00168371,    13.8026323,   -6.67062282,
      9.94362449,    -4.6888504,    -9.90709114,   7.902920e-01, -10.5181952,
      2.20467281,    -15.2523861,   -38.8545609,   11.9189167,   -2.62887621,
      8.58652305,    -0.473736167,  7.1090169,     -3.85814524,  8.66619777,
      2.62849188,    -6.02289867,   38.5086594,    -14.960989,   -0.931303263,
      -1.54379117,   18.820015,     11.8206406,    -20.333456,   -7.89252043,
      29.008894,     -25.7186718,   -19.4724236,   30.11619,     7.93340731,
      -16.3272209,   -15.2483006,   17.3349628,    10.0789604,   -7.93241549,
      -15.3392467,   16.5936661,    2.55696344,    -27.9144192,  7.80609798,
      -3.229860e+01, 25.9569187,    4.88266182,    16.6617928,   42.6433372,
      -2.0199244,    -16.1016083,   -4.36955547,   22.9223862,   10.6770782,
      31.7341576,    13.8382187,    -6.103100e+00, 14.9103413,   19.7866287,
      -5.49546337,   -13.8123255,   -5.622370e+00, -4.382442,    -4.5823946,
      -13.823595,    -8.07380295,   22.0516758,    -15.5325346,  19.3795795,
      4.62363148,    -42.9268227,   44.568615,     -25.3516674,  25.9454765,
      -1.98024702,   -21.8099804,   -5.91219329,   1.73632288,   7.70343351,
      0.479099929,   14.0837402,    5.74589205,    19.2407303,   -0.745551348,
      7.30496501,    -2.79473114,   -14.1908264,   -0.069957979, 13.4065199,
      -25.4774895,   0.87921524,    -7.00054693,   -21.0068073,  -20.6663818,
      -13.3254213,   5.31242561,    1.70617437,    -24.5330563,  4.035820e+00,
      -4.91719818,   6.942873,      13.9773598,    -3.05933499,  -28.627018,
      13.0832052,    16.920908,     -20.4616985,   -5.82142353,  -1.07573605,
      -55.1605568,   31.1032867,    38.8496704,    -18.9358845,  53.0218925,
      6.82090235,    -17.4631557,   -6.658140e+00, -49.0257568,  -57.965126,
      1.28888094,    10.1064234,    11.615819,     6.30278158,   5.33166265,
      5.80028343,    6.36568213,    -23.0659161,   8.62879085,   6.13341904,
      -8.5325222,    6.88468885,    -10.2892427,   -27.1238651,  -1.03788805,
      -0.98752737,   0.675509929,   -21.9212456,   11.298914,    -29.9368305,
      13.2601376,    -6.06431961,   -9.05289554,   12.9358139,   8.66670227,
      0.0313370228,  -25.177578,    -23.8081703,   4.7123723,    -8.822750e+00,
      -6.71417427,   -11.3798552,   2.39726067,    6.18268871,   0.0120680928,
      4.38614082,    10.0917692,    -40.2899895,   -4.36912107,  -4.1604104,
      -3.29209113,   -1.90671635,   -1.83079815,   -12.7707281,  3.1059351,
      -12.731389,    14.3053932,    -1.6831131,    -14.3278656,  5.57291031,
      -20.3143883,   60.1191406,    -36.1097336,   -4.75165796,  -8.16517353,
      21.6081047,    4.701960e+01,  -51.0683784,   1.03911972,   -29.8507824,
      0.616962552,   -4.04210424,   3.10792017,    0.520172954,  -0.645381271,
      0.14783594,    -15.554266,    18.4245739,    6.26318932,   10.3815079,
      6.32173061,    -23.0389137,   -33.8552246,   0.677015125,  -4.145770e+00,
      22.203455,     -1.02386034,   -5.320930e+00, -19.5210209,  -20.0154915,
      1.88341713,    -6.25665617,   30.7729092,    2.85177326,   -24.6818047,
      -45.2080078,   -6.05969381,   -27.0595455,   -4.84107685,  -13.6823683,
      -0.22210598,   -37.4834099,   0.53820014,    -23.2549763,  25.7111702,
      -9.00931549,   1.49701595,    -5.79211712,   0.189515129,  2.36139631,
      -0.109550804,  -5.44110584,   6.62728786,    1.61547112,   16.660017,
      -18.7215042,   -22.5491428,   9.18865966,    7.77150154,   -14.6802511,
      -6.93513774,   5.96822214,    -12.4512501,   14.6568079,   -30.7445717,
      25.5005951,    11.6696749,    -37.512207,    1.76274657,   17.9486141,
      -11.7353954,   -0.0957496166, 0.0172402859,  -10.5637932,  -38.4266739,
      0.462153196,   -6.09720611,   6.67712259,    55.8793182,   5.73991585,
      -9.39302825,   -4.38571358,   -10.4928865,   -0.766156673, -18.3599625,
      -11.7640486,   17.1294804,    -11.8793659,   1.62921071,   0.12774086,
      -20.1454811,   4.94842529,    -11.9677525,   -9.4519186,   2.14683056,
      -8.7701826,    -16.4303188,   -35.3038368,   -43.4693451,  -61.954071,
      -6.31224918,   -6.64896727,   -25.0835743,   -5.23444939,  -17.669714,
      -1.74165297,   6.51037025,    -29.6431427,   -37.8414307,  4.869830e+00,
      6.90403318,    -13.2158976,   -10.0932417,   -12.3609695,  5.36755276,
      2.04900503,    2.12651753,    -16.7633343};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork4) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2});
  const Shape shape_rhs({2, 2, 2});
  const Shape shape_r({2, 2, 2});

  Vector<StorageT> lhs_data{1.1, 2.2, 3.3, 4.3, 5.5, 6, 7, 8};
  Vector<StorageT> rhs_data{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {1.32, 2.64, 3.96, 5.16, 6.6, 7.2, 8.4, 9.6};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32DotGeneralTest, kF32TestTypesTensorsWork5) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4});
  const Shape shape_rhs({4, 4});
  const Shape shape_r({4});

  Vector<StorageT> lhs_data{-2.67958117, 4.91505384, -2.93944049, -0.189866632};
  Vector<StorageT> rhs_data{
      -0.876051664, 2.81679201, 1.48077691,   1.10807765,
      -1.83372617,  1.35355616, 3.68328929,   -4.30171204,
      -6.15009593,  -5.9722824, -0.454436153, -1.66895545,
      1.09934378,   5.87006092, -3.10807371,  0.333222806};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{0};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data_float = {11.203701, 15.5456181, 16.0616112,
                                          -19.2698021};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data_float));
}

using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
template <class T>
struct NonQuantizedkBF16DotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkBF16DotGeneralTest, kBF16TestTypes,
                 TestParamNames);

TYPED_TEST(NonQuantizedkBF16DotGeneralTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({3});
  const Shape shape_rhs({3});
  const Shape shape_r({1});

  Vector<float> lhs_data_float{-3.859380, 3.109380, -3.000000};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{8.476560e-01, -3.171880e+00, 8.984370e-01};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{0};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data = {StorageT(-15.8292847)};

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkBF16DotGeneralTest, kBF16TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 3, 4});
  const Shape shape_rhs({1, 4, 3});
  const Shape shape_r({1});

  Vector<float> lhs_data_float{4.968750e+00,  -1.101560e+00, 1.015630e+00,
                               4.812500e+00,  -3.398440e-01, 2.484380e+00,
                               -5.187500e+00, -1.109380e+00, -1.328130e+00,
                               3.312500e+00,  -4.937500e+00, -4.281250e+00};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{-1.164060e+00, 1.437500e+00,  -4.638670e-02,
                               -1.945310e+00, -5.187500e+00, -1.414060e+00,
                               -2.031250e+00, -3.656250e+00, -1.738280e-01,
                               4.902340e-01,  -5.968750e+00, -3.671880e+00};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2, 1};
  Vector<Axis> rhsc_dim{1, 2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data_float{2.087500e+01};
  Vector<StorageT> expected_data(expected_data_float.begin(),
                                 expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
template <class T>
struct NonQuantizedkF16DotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF16DotGeneralTest, kF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF16DotGeneralTest, kF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({2, 2, 2, 2});
  const Shape shape_rhs({2, 2, 2, 2});
  const Shape shape_r({2, 2, 2, 2});

  Vector<float> lhs_data_float{1.1,  2.2,   3.3,   4.3,   5.5,   6,   7,   8,
                               11.1, 12.22, 33.33, 44.32, 15.15, 6.6, 7.3, 8.1};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2,
                               1.2, 0, 0, 1.2, 1.2, 0, 0, 1.2};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0, 3};
  Vector<Axis> rhsb_dim{0, 3};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {
      1.319, 1.319, 6.6,   6.6,   5.16,   5.16,   9.6,    9.6,
      13.32, 13.32, 18.18, 18.18, 53.184, 53.184, 9.7265, 9.7265};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF16DotGeneralTest, kF16TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({4, 4});
  const Shape shape_rhs({4});
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
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {80.250000, -8.242180e-01, -4.181250e+01,
                                       2.057810e+01};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedIntDotGeneralTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedIntDotGeneralTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({1, 2, 2});
  const Shape shape_rhs({1, 2, 2});
  const Shape shape_r({1, 2, 2});

  Vector<StorageT> lhs_data = Vector<StorageT>{10, 8, 1, 2};
  Vector<StorageT> rhs_data = Vector<StorageT>{1, 0, 1, 1};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  const ExpressedT scale = static_cast<ExpressedT>(2);
  const StorageT zero_point = static_cast<StorageT>(0);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{.type = QuantizedPerTensorTensorType{.shape = shape_rhs,
                                                  .element_type = tensor_type},
             .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data = Vector<float>{44, 4, 40, 8};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({4, 3});
  const Shape shape_rhs({3});
  const Shape shape_r({4});

  Vector<StorageT> lhs_data =
      Vector<StorageT>{0, 0, 2, 0, 1, 2, 4, 2, 0, 1, 2, 6};
  Vector<StorageT> rhs_data = Vector<StorageT>{1, 1, 0};
  Vector<Axis> lhsb_dim{};
  Vector<Axis> rhsb_dim{};
  Vector<Axis> lhsc_dim{1};
  Vector<Axis> rhsc_dim{0};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  Vector<StorageT> output_data(shape_r.NumElements());

  const ExpressedT scale = static_cast<ExpressedT>(1.2);
  const StorageT zero_point = static_cast<StorageT>(-1);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  const QuantizedElementTypePerTensor tensor_type_rhs =
      QuantizedElementTypePerTensor(TypeParam::kStorage, 0,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerTensorTensorType{.shape = shape_rhs,
                                           .element_type = tensor_type_rhs},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data = Vector<float>{2.88, 4.32, 11.531, 7.2};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({2, 2, 2});
  const Shape shape_rhs({2, 2, 2});
  const Shape shape_r({2, 2, 2});

  Vector<StorageT> lhs_data = Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8};
  Vector<StorageT> rhs_data = Vector<StorageT>{2, 0, 0, 2, 2, 0, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);
  const ExpressedT scale = static_cast<ExpressedT>(1.3);
  const StorageT zero_point = static_cast<StorageT>(0);
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.7, 1.6};

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 2);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerAxisTensorType{.shape = shape_rhs,
                                         .element_type = tensor_type_axis},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data =
      Vector<float>{4.417, 8.32, 13.257, 16.64, 22.109, 24.937, 30.953, 33.281};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork4) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({2, 2, 2});
  const Shape shape_rhs({2, 2, 2});
  const Shape shape_r({2, 2, 2});

  Vector<StorageT> lhs_data = Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8};
  Vector<StorageT> rhs_data = Vector<StorageT>{2, 0, 0, 2, 2, 0, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2};
  Vector<Axis> rhsc_dim{1};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  const ExpressedT scale = static_cast<ExpressedT>(1.4);
  const StorageT zero_point = static_cast<StorageT>(0);
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.7, 1.6};
  Vector<int> zeroes = {0, 0};
  Vector<float> scalesv = {1.7, 1.6};

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 2);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerAxisTensorType{.shape = shape_rhs,
                                         .element_type = tensor_type_axis},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerAxisTensorType{.shape = shape_r,
                                         .element_type = tensor_type_axis},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data = {4.76172, 8.96094, 14.289, 17.921,
                                 23.796,  26.89,   33.34,  35.843};

  Vector<float> quantized_data(shape_r.NumElements());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    int quantization_index = i % 2;
    StorageT quantized_value =
        Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            static_cast<ExpressedT>(expected_data[i]),
            zeroes[quantization_index],
            static_cast<ExpressedT>(1.0f / scalesv[quantization_index]));
    quantized_data[i] = quantized_value;
  }

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), quantized_data));
}

TYPED_TEST(QuantizedIntDotGeneralTest, QuantizedTestTypesTensorsWork5) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({1, 3, 4});
  const Shape shape_rhs({1, 4, 3});
  const Shape shape_r({1});

  Vector<StorageT> lhs_data{2, 0, 0, 0, 5, -3, 0, 4, -1, 0, 0, -1};
  Vector<StorageT> rhs_data{0, 4, 2, 3, 3, 3, -6, -2, 1, -1, 1, 0};
  Vector<StorageT> output_data(shape_r.NumElements());
  Vector<Axis> lhsb_dim{0};
  Vector<Axis> rhsb_dim{0};
  Vector<Axis> lhsc_dim{2, 1};
  Vector<Axis> rhsc_dim{1, 2};
  absl::Span<const Axis> lhs_batching_dimensions(lhsb_dim);
  absl::Span<const Axis> rhs_batching_dimensions(rhsb_dim);
  absl::Span<const Axis> lhs_contracting_dimensions(lhsc_dim);
  absl::Span<const Axis> rhs_contracting_dimensions(rhsc_dim);

  const ExpressedT scale = static_cast<ExpressedT>(1.4);
  const StorageT zero_point = static_cast<StorageT>(0);
  std::initializer_list<float> zero_points = {0};
  std::initializer_list<float> scales = {1.2};
  std::vector<int> zeroes = {0};
  std::vector<float> scalesv = {1.2};

  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerAxisTensorType{.shape = shape_rhs,
                                         .element_type = tensor_type_axis},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerAxisTensorType{.shape = shape_r,
                                         .element_type = tensor_type_axis},
      .data = output_data.data()};
  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  Vector<float> expected_data{21.840004};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zeroes[0],
                       static_cast<ExpressedT>(1.0 / scalesv[0]));
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

}  // namespace
}  // namespace shlo_ref