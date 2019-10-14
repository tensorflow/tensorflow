/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/tridiagonal.h"

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace tridiagonal {
namespace {

class TridiagonalTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

XLA_TEST_P(TridiagonalTest, Solves) {
  const auto& spec = GetParam();
  xla::XlaBuilder builder(TestName());

  const int64 num_eqs = 5;
  const int64 num_rhs = 3;
  const int64 lower_diagonal_batch_size = std::get<0>(spec);
  const int64 main_diagonal_batch_size = std::get<1>(spec);
  const int64 upper_diagonal_batch_size = std::get<2>(spec);
  const int64 rhs_diagonal_batch_size = std::get<2>(spec);

  const int64 max_batch_size =
      std::max({lower_diagonal_batch_size, main_diagonal_batch_size,
                upper_diagonal_batch_size, rhs_diagonal_batch_size});

  Array3D<float> lower_diagonal(lower_diagonal_batch_size, 1, num_eqs);
  Array3D<float> main_diagonal(main_diagonal_batch_size, 1, num_eqs);
  Array3D<float> upper_diagonal(upper_diagonal_batch_size, 1, num_eqs);
  Array3D<float> rhs(rhs_diagonal_batch_size, num_rhs, num_eqs);

  lower_diagonal.FillRandom(1.0, /*mean=*/0.0, /*seed=*/0);
  main_diagonal.FillRandom(0.05, /*mean=*/1.0,
                           /*seed=*/max_batch_size * num_eqs);
  upper_diagonal.FillRandom(1.0, /*mean=*/0.0,
                            /*seed=*/2 * max_batch_size * num_eqs);
  rhs.FillRandom(1.0, /*mean=*/0.0, /*seed=*/3 * max_batch_size * num_eqs);

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

  TF_ASSERT_OK_AND_ASSIGN(XlaOp x,
                          ThomasSolver(lower_diagonal_xla, main_diagonal_xla,
                                       upper_diagonal_xla, rhs_xla));

  auto Coefficient = [](auto operand, auto i) {
    return SliceInMinorDims(operand, /*start=*/{i}, /*end=*/{i + 1});
  };

  std::vector<XlaOp> relative_errors(num_eqs);

  for (int64 i = 0; i < num_eqs; i++) {
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
                        ::testing::Combine(::testing::Values(1, 8),
                                           ::testing::Values(1, 8),
                                           ::testing::Values(1, 8),
                                           ::testing::Values(1, 8)));

}  // namespace
}  // namespace tridiagonal
}  // namespace xla
