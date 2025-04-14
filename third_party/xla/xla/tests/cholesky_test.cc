/* Copyright 2018 The OpenXLA Authors.

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

#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/array2d.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"

namespace xla {
namespace {

using CholeskyTest = ClientLibraryTestBase;

TEST_F(CholeskyTest, NonPSDInput) {
  XlaBuilder builder(TestName());

  Array2D<float> a_vals({
      {1, 1, 1},
      {1, 1, 1},
      {1, 1, 1},
  });

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  Cholesky(a, /*lower=*/true);

  float nan = std::numeric_limits<float>::quiet_NaN();
  Array2D<float> expected({
      {nan, nan, nan},
      {nan, nan, nan},
      {nan, nan, nan},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(CholeskyTest, NonPSDBatched) {
  XlaBuilder builder(TestName());

  Array3D<float> a_vals({
      {
          {10, 0, 0},
          {1, 20, 0},
          {1, 1, 30},
      },
      {
          {1, 1, 1},
          {1, 1, 1},
          {1, 1, 1},
      },
  });

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  Cholesky(a, /*lower=*/true);

  float nan = std::numeric_limits<float>::quiet_NaN();
  Array3D<float> expected({
      {
          {3.16227766, 0., 0.},
          {0.31622777, 4.4609416, 0.},
          {0.31622777, 0.20175113, 5.46436606},
      },
      {
          {nan, nan, nan},
          {nan, nan, nan},
          {nan, nan, nan},
      },
  });

  ComputeAndCompareR3<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(CholeskyTest, Lower) {
  XlaBuilder builder(TestName());

  float nan = std::numeric_limits<float>::quiet_NaN();
  Array2D<float> a_vals({
      {4, nan, nan, nan},
      {6, 45, nan, nan},
      {8, 54, 146, nan},
      {10, 63, 166, 310},
  });

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  LowerTriangle(Cholesky(a, /*lower=*/true));

  Array2D<float> expected({
      {2, 0, 0, 0},
      {3, 6, 0, 0},
      {4, 7, 9, 0},
      {5, 8, 10, 11},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(CholeskyTest, Upper) {
  XlaBuilder builder(TestName());

  float nan = std::numeric_limits<float>::quiet_NaN();
  Array2D<float> a_vals({
      {4, 6, 8, 10},
      {nan, 45, 54, 63},
      {nan, nan, 146, 166},
      {nan, nan, nan, 310},
  });

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  UpperTriangle(Cholesky(a, /*lower=*/false));

  Array2D<float> expected({
      {2, 3, 4, 5},
      {0, 6, 7, 8},
      {0, 0, 9, 10},
      {0, 0, 0, 11},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(CholeskyTest, Simple2) {
  XlaBuilder builder(TestName());

  Array2D<float> a_vals({
      {16, 24, 8, 12},
      {24, 61, 82, 48},
      {8, 82, 456, 106},
      {12, 48, 106, 62},
  });

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  LowerTriangle(Cholesky(a, /*lower=*/true));

  Array2D<float> expected({{4, 0, 0, 0},    //
                           {6, 5, 0, 0},    //
                           {2, 14, 16, 0},  //
                           {3, 6, 1, 4}});

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(CholeskyTest, SimpleBatched) {
  XlaBuilder builder(TestName());

  Array3D<float> a_vals({
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

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  LowerTriangle(Cholesky(a, /*lower=*/true));

  Array3D<float> expected({
      {
          {2, 0, 0, 0},
          {3, 6, 0, 0},
          {4, 7, 9, 0},
          {5, 8, 10, 11},
      },
      {{4, 0, 0, 0},    //
       {6, 5, 0, 0},    //
       {2, 14, 16, 0},  //
       {3, 6, 1, 4}},
  });

  ComputeAndCompareR3<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

using CholeskyTestCase = std::tuple<int64_t, int64_t, bool>;

class RandomCholeskyTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<CholeskyTestCase> {};

TEST_P(RandomCholeskyTest, Real) {
  XlaBuilder builder(TestName());

  auto test_params = GetParam();
  std::vector<int64_t> dimensions = {std::get<0>(test_params),
                                     std::get<1>(test_params),
                                     std::get<1>(test_params)};
  bool lower = std::get<2>(test_params);
  Shape shape = ShapeUtil::MakeShape(F32, dimensions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto literal, LiteralUtil::CreateRandomLiteral<F32>(shape, 0.0, 1.0));

  auto input = Parameter(&builder, 0, shape, "input");
  // Form a random positive definite matrix.
  auto matrix =
      BatchDot(input, TransposeInMinorDims(input), PrecisionConfig::HIGHEST);

  auto cholesky = Triangle(Cholesky(matrix, lower), lower);

  // Verify that ||matrix - cholesky * cholesky_t||_2 ~= 0
  XlaOp verification;
  if (lower) {
    verification = BatchDot(cholesky, TransposeInMinorDims(cholesky),
                            PrecisionConfig::HIGHEST);
  } else {
    verification = BatchDot(TransposeInMinorDims(cholesky), cholesky,
                            PrecisionConfig::HIGHEST);
  }
  auto delta = matrix - verification;
  Reduce(delta * delta, ConstantR0<float>(&builder, 0.0),
         CreateScalarAddComputation(F32, &builder), {0, 1, 2});

  TF_ASSERT_OK_AND_ASSIGN(auto input_data, client_->TransferToServer(literal));
  ComputeAndCompareR0<float>(&builder, 0.0, {input_data.get()},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_P(RandomCholeskyTest, Complex) {
  XlaBuilder builder(TestName());

  auto test_params = GetParam();
  std::vector<int64_t> dimensions = {std::get<0>(test_params),
                                     std::get<1>(test_params),
                                     std::get<1>(test_params)};
  bool lower = std::get<2>(test_params);
  Shape shape = ShapeUtil::MakeShape(F32, dimensions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto literal_real,
      LiteralUtil::CreateRandomLiteral<F32>(shape, 0.0, 1.0));
  TF_ASSERT_OK_AND_ASSIGN(
      auto literal_imag,
      LiteralUtil::CreateRandomLiteral<F32>(shape, 0.0, 1.0));

  auto input_real = Parameter(&builder, 0, shape, "input_real");
  auto input_imag = Parameter(&builder, 1, shape, "input_imag");
  auto input = Complex(input_real, input_imag);
  // Form a random positive definite matrix.
  auto matrix = BatchDot(input, TransposeInMinorDims(Conj(input)),
                         PrecisionConfig::HIGHEST);

  auto cholesky = Triangle(Cholesky(matrix, lower), lower);

  // Verify that ||matrix - cholesky * cholesky_t||_2 ~= 0
  XlaOp verification;
  if (lower) {
    verification = BatchDot(cholesky, TransposeInMinorDims(Conj(cholesky)),
                            PrecisionConfig::HIGHEST);
  } else {
    verification = BatchDot(TransposeInMinorDims(Conj(cholesky)), cholesky,
                            PrecisionConfig::HIGHEST);
  }
  auto delta = matrix - verification;
  Reduce(Abs(delta * Conj(delta)), ConstantR0<float>(&builder, 0.0),
         CreateScalarAddComputation(F32, &builder), {0, 1, 2});

  TF_ASSERT_OK_AND_ASSIGN(auto input_data_real,
                          client_->TransferToServer(literal_real));
  TF_ASSERT_OK_AND_ASSIGN(auto input_data_imag,
                          client_->TransferToServer(literal_imag));
  ComputeAndCompareR0<float>(&builder, 0.0,
                             {input_data_real.get(), input_data_imag.get()},
                             ErrorSpec(1e-4, 1e-4));
}

INSTANTIATE_TEST_SUITE_P(RandomCholeskyTestInstance, RandomCholeskyTest,
                         ::testing::Values(CholeskyTestCase{1, 1, true},
                                           CholeskyTestCase{1, 2, true},
                                           CholeskyTestCase{1, 50, true},
                                           CholeskyTestCase{1, 50, false},
                                           CholeskyTestCase{1, 255, false},
                                           CholeskyTestCase{10, 5, true},
                                           CholeskyTestCase{5, 10, false},
                                           CholeskyTestCase{2, 20, true},
                                           CholeskyTestCase{2, 129, true}));

}  // namespace
}  // namespace xla
