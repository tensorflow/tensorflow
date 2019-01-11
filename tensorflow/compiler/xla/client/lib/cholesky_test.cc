/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/cholesky.h"

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

using xla::int64;

using CholeskyTest = xla::ClientLibraryTestBase;

XLA_TEST_F(CholeskyTest, Simple) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::Cholesky(a, /*block_size=*/2);

  xla::Array2D<float> expected({
      {2, 0, 0, 0},
      {3, 6, 0, 0},
      {4, 7, 9, 0},
      {5, 8, 10, 11},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(CholeskyTest, Simple2) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {16, 24, 8, 12},
      {24, 61, 82, 48},
      {8, 82, 456, 106},
      {12, 48, 106, 62},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::Cholesky(a);

  xla::Array2D<float> expected(
      {{4, 0, 0, 0}, {6, 5, 0, 0}, {2, 14, 16, 0}, {3, 6, 1, 4}});

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(CholeskyTest, SimpleBatched) {
  xla::XlaBuilder builder(TestName());

  xla::Array3D<float> a_vals({
      {
          {4, 6, 8, 10},
          {6, 45, 54, 63},
          {8, 54, 146, 166},
          {10, 63, 166, 310},
      },
      {
          {16, 24, 8, 12},
          {24, 61, 82, 48},
          {8, 82, 456, 106},
          {12, 48, 106, 62},
      },
  });

  xla::XlaOp a;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::Cholesky(a);

  xla::Array3D<float> expected({
      {
          {2, 0, 0, 0},
          {3, 6, 0, 0},
          {4, 7, 9, 0},
          {5, 8, 10, 11},
      },
      {{4, 0, 0, 0}, {6, 5, 0, 0}, {2, 14, 16, 0}, {3, 6, 1, 4}},
  });

  ComputeAndCompareR3<float>(&builder, expected, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

using CholeskyTestCase = std::tuple<int64, int64>;

class RandomCholeskyTest
    : public xla::ClientLibraryTestBase,
      public ::testing::WithParamInterface<CholeskyTestCase> {};

XLA_TEST_P(RandomCholeskyTest, Random) {
  xla::XlaBuilder builder(TestName());

  auto test_params = GetParam();
  std::vector<int64> dimensions = {std::get<0>(test_params),
                                   std::get<1>(test_params),
                                   std::get<1>(test_params)};
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, dimensions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto literal,
      xla::LiteralUtil::CreateRandomLiteral<xla::F32>(shape, 0.0, 1.0));

  auto input = xla::Parameter(&builder, 0, shape, "input");
  // Form a random positive definite matrix.
  auto matrix = xla::BatchDot(input, TransposeInMinorDims(input),
                              xla::PrecisionConfig::HIGHEST);

  auto cholesky = xla::Cholesky(matrix, /*block_size=*/4);

  // Verify that ||matrix - cholesky * cholesky_t||_2 ~= 0
  auto verification = xla::BatchDot(cholesky, TransposeInMinorDims(cholesky),
                                    xla::PrecisionConfig::HIGHEST);
  auto delta = matrix - verification;
  xla::Reduce(delta * delta, xla::ConstantR0<float>(&builder, 0.0),
              CreateScalarAddComputation(xla::F32, &builder), {0, 1, 2});

  TF_ASSERT_OK_AND_ASSIGN(auto input_data, client_->TransferToServer(literal));
  ComputeAndCompareR0<float>(&builder, 0.0, {input_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

INSTANTIATE_TEST_CASE_P(RandomCholeskyTestInstance, RandomCholeskyTest,
                        ::testing::Values(CholeskyTestCase{1, 1},
                                          CholeskyTestCase{1, 2},
                                          CholeskyTestCase{10, 5},
                                          CholeskyTestCase{2, 20}));

}  // namespace
