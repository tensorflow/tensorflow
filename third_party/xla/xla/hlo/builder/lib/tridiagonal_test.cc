/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/tridiagonal.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/array.h"
#include "xla/array3d.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace tridiagonal {
namespace {

class TridiagonalTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

XLA_TEST_P(TridiagonalTest, SimpleTridiagonalMatMulOk) {
  xla::XlaBuilder builder(TestName());

  // Since the last element ignored, it will be {{{34, 35, 0}}}
  Array3D<float> upper_diagonal{{{34, 35, 999}}};
  Array3D<float> main_diagonal{{{21, 22, 23}}};
  // Since the first element ignored, it will be {{{0, 10, 100}}}
  Array3D<float> lower_diagonal{{{999, 10, 100}}};
  Array3D<float> rhs{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}};

  XlaOp upper_diagonal_xla;
  XlaOp main_diagonal_xla;
  XlaOp lower_diagonal_xla;
  XlaOp rhs_xla;

  auto upper_diagonal_data = CreateR3Parameter<float>(
      upper_diagonal, 0, "upper_diagonal", &builder, &upper_diagonal_xla);
  auto main_diagonal_data = CreateR3Parameter<float>(
      main_diagonal, 1, "main_diagonal", &builder, &main_diagonal_xla);
  auto lower_diagonal_data = CreateR3Parameter<float>(
      lower_diagonal, 2, "lower_diagonal", &builder, &lower_diagonal_xla);
  auto rhs_data = CreateR3Parameter<float>(rhs, 3, "rhs", &builder, &rhs_xla);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaOp x, TridiagonalMatMul(upper_diagonal_xla, main_diagonal_xla,
                                 lower_diagonal_xla, rhs_xla));

  ASSERT_EQ(x.builder()->first_error(), absl::OkStatus());
  ASSERT_TRUE(x.valid());

  std::vector<int64_t> expected_shape{1, 3, 4};
  std::vector<float> expected_values{191, 246, 301, 356, 435, 502,
                                     569, 636, 707, 830, 953, 1076};
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      ComputeAndTransfer(x.builder(),
                         {upper_diagonal_data.get(), main_diagonal_data.get(),
                          lower_diagonal_data.get(), rhs_data.get()}));
  EXPECT_EQ(result.shape().dimensions(), expected_shape);
  EXPECT_EQ(result.data<float>({}), expected_values);
}

XLA_TEST_P(TridiagonalTest, TridiagonalMatMulWrongShape) {
  xla::XlaBuilder builder(TestName());

  Array<float> upper_diagonal = Array<float>({5, 3, 7}, 1);
  Array<float> main_diagonal = Array<float>({5, 3, 7}, 1);
  Array<float> lower_diagonal = Array<float>({5, 3, 7}, 1);
  Array<float> rhs = Array<float>({5, 3, 7, 6}, 1);

  XlaOp upper_diagonal_xla;
  XlaOp main_diagonal_xla;
  XlaOp lower_diagonal_xla;
  XlaOp rhs_xla;

  auto upper_diagonal_data = CreateParameter<float>(
      upper_diagonal, 0, "upper_diagonal", &builder, &upper_diagonal_xla);
  auto main_diagonal_data = CreateParameter<float>(
      main_diagonal, 1, "main_diagonal", &builder, &main_diagonal_xla);
  auto lower_diagonal_data = CreateParameter<float>(
      lower_diagonal, 2, "lower_diagonal", &builder, &lower_diagonal_xla);
  auto rhs_data = CreateParameter<float>(rhs, 3, "rhs", &builder, &rhs_xla);

  auto result = TridiagonalMatMul(upper_diagonal_xla, main_diagonal_xla,
                                  lower_diagonal_xla, rhs_xla);
  ASSERT_EQ(result.status(),
            InvalidArgument(
                "superdiag must have same rank as rhs, but got 3 and 4."));
}

XLA_TEST_P(TridiagonalTest, Solves) {
  const auto& spec = GetParam();
  xla::XlaBuilder builder(TestName());

  // TODO(belletti): parametrize num_rhs.
  const int64_t batch_size = std::get<0>(spec);
  const int64_t num_eqs = std::get<1>(spec);
  const int64_t num_rhs = std::get<2>(spec);

  Array3D<float> lower_diagonal(batch_size, 1, num_eqs);
  Array3D<float> main_diagonal(batch_size, 1, num_eqs);
  Array3D<float> upper_diagonal(batch_size, 1, num_eqs);
  Array3D<float> rhs(batch_size, num_rhs, num_eqs);

  lower_diagonal.FillRandom(1.0, /*mean=*/0.0, /*seed=*/0);
  main_diagonal.FillRandom(0.05, /*mean=*/1.0,
                           /*seed=*/batch_size * num_eqs);
  upper_diagonal.FillRandom(1.0, /*mean=*/0.0,
                            /*seed=*/2 * batch_size * num_eqs);
  rhs.FillRandom(1.0, /*mean=*/0.0, /*seed=*/3 * batch_size * num_eqs);

  XlaOp lower_diagonal_xla;
  XlaOp main_diagonal_xla;
  XlaOp upper_diagonal_xla;
  XlaOp rhs_xla;

  auto lower_diagonal_data = CreateR3Parameter<float>(
      lower_diagonal, 0, "lower_diagonal", &builder, &lower_diagonal_xla);
  auto main_diagonal_data = CreateR3Parameter<float>(
      main_diagonal, 1, "main_diagonal", &builder, &main_diagonal_xla);
  auto upper_diagonal_data = CreateR3Parameter<float>(
      upper_diagonal, 2, "upper_diagonal", &builder, &upper_diagonal_xla);
  auto rhs_data = CreateR3Parameter<float>(rhs, 3, "rhs", &builder, &rhs_xla);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaOp x, TridiagonalSolver(kThomas, lower_diagonal_xla, main_diagonal_xla,
                                 upper_diagonal_xla, rhs_xla));

  auto Coefficient = [](auto operand, auto i) {
    return SliceInMinorDims(operand, /*start=*/{i}, /*end=*/{i + 1});
  };

  std::vector<XlaOp> relative_errors(num_eqs);

  for (int64_t i = 0; i < num_eqs; i++) {
    auto a_i = Coefficient(lower_diagonal_xla, i);
    auto b_i = Coefficient(main_diagonal_xla, i);
    auto c_i = Coefficient(upper_diagonal_xla, i);
    auto d_i = Coefficient(rhs_xla, i);

    if (i == 0) {
      relative_errors[i] =
          (b_i * Coefficient(x, i) + c_i * Coefficient(x, i + 1) - d_i) / d_i;
    } else if (i == num_eqs - 1) {
      relative_errors[i] =
          (a_i * Coefficient(x, i - 1) + b_i * Coefficient(x, i) - d_i) / d_i;
    } else {
      relative_errors[i] =
          (a_i * Coefficient(x, i - 1) + b_i * Coefficient(x, i) +
           c_i * Coefficient(x, i + 1) - d_i) /
          d_i;
    }
  }
  Abs(ConcatInDim(&builder, relative_errors, 2));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      ComputeAndTransfer(&builder,
                         {lower_diagonal_data.get(), main_diagonal_data.get(),
                          upper_diagonal_data.get(), rhs_data.get()}));

  auto result_data = result.data<float>({});
  for (auto result_component : result_data) {
    EXPECT_TRUE(result_component < 5e-3);
  }
}

INSTANTIATE_TEST_CASE_P(TridiagonalTestInstantiation, TridiagonalTest,
                        ::testing::Combine(::testing::Values(1, 12),
                                           ::testing::Values(4, 8),
                                           ::testing::Values(1, 12)));

}  // namespace
}  // namespace tridiagonal
}  // namespace xla
