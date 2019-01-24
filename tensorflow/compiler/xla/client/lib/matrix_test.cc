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

#include "tensorflow/compiler/xla/client/lib/matrix.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class MatrixTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  void TestMatrixDiagonal();
};

XLA_TEST_F(MatrixTest, Triangle) {
  XlaBuilder builder(TestName());
  Array3D<int32> input(2, 3, 4);
  input.FillIota(0);

  XlaOp a;
  auto a_data = CreateR3Parameter<int32>(input, 0, "a", &builder, &a);
  LowerTriangle(a);
  Array3D<int32> expected({{{0, 0, 0, 0}, {4, 5, 0, 0}, {8, 9, 10, 0}},
                           {{12, 0, 0, 0}, {16, 17, 0, 0}, {20, 21, 22, 0}}});

  ComputeAndCompareR3<int32>(&builder, expected, {a_data.get()});
}

template <typename T>
void MatrixTest::TestMatrixDiagonal() {
  XlaBuilder builder("GetMatrixDiagonal");
  Array3D<T> input(2, 3, 4);
  input.FillIota(0);

  XlaOp a;
  auto a_data = CreateR3Parameter<T>(input, 0, "a", &builder, &a);
  GetMatrixDiagonal(a);
  Array2D<T> expected({{0, 5, 10}, {12, 17, 22}});

  ComputeAndCompareR2<T>(&builder, expected, {a_data.get()});
}

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_S32) { TestMatrixDiagonal<int32>(); }

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_S64) { TestMatrixDiagonal<int64>(); }

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_F32) { TestMatrixDiagonal<float>(); }

Array3D<float> BatchedAValsFull() {
  return {{
              {2, 0, 1, 2},
              {3, 6, 0, 1},
              {4, 7, 9, 0},
              {5, 8, 10, 11},
          },
          {
              {16, 24, 8, 12},
              {24, 61, 82, 48},
              {8, 82, 456, 106},
              {12, 48, 106, 62},
          }};
}

XLA_TEST_F(MatrixTest, RowBatchDot) {
  XlaBuilder builder(TestName());

  int n = 4;

  XlaOp a, row, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto row_data = CreateR3Parameter<float>({{{9, 1, 0, 0}}, {{2, 4, 0, 0}}}, 1,
                                           "row", &builder, &row);
  // Select {{3, 6, 0, 1}, {24, 61,  82,  48}} out of BatchedAValsFull().
  auto index_data = CreateR0Parameter<int>(1, 2, "index", &builder, &index);

  auto l_index = DynamicSliceInMinorDims(
      a, {index, ConstantR0<int32>(&builder, 0)}, {1, n});
  BatchDot(l_index, TransposeInMinorDims(row));

  ComputeAndCompareR3<float>(&builder, {{{33}}, {{292}}},
                             {a_data.get(), row_data.get(), index_data.get()});
}

XLA_TEST_F(MatrixTest, Einsum) {
  XlaBuilder builder(TestName());

  int n = 4;

  XlaOp a, row, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto row_data = CreateR3Parameter<float>({{{9, 1, 0, 0}}, {{2, 4, 0, 0}}}, 1,
                                           "row", &builder, &row);
  // Select {{3, 6, 0, 1}, {24, 61,  82,  48}} out of BatchedAValsFull().
  auto index_data = CreateR0Parameter<int>(1, 2, "index", &builder, &index);

  auto l_index = DynamicSliceInMinorDims(
      a, {index, ConstantR0<int32>(&builder, 0)}, {1, n});
  Einsum(l_index, row, "abc,adc->abd");

  ComputeAndCompareR3<float>(&builder, {{{33}}, {{292}}},
                             {a_data.get(), row_data.get(), index_data.get()});
}

XLA_TEST_F(MatrixTest, ParseEinsumString) {
  auto to_vec = [](absl::string_view s) {
    std::vector<int64> v;
    v.reserve(s.size());
    for (auto c : s) {
      v.push_back(int64{c});
    }
    return v;
  };

  auto to_string = [&](absl::string_view x, absl::string_view y,
                       absl::string_view o) {
    return absl::StrCat(x, ",", y, "->", o);
  };

  std::vector<std::vector<string>> good_test_cases = {{"ab", "bc", "ac"},
                                                      {"Bab", "Bbc", "Bac"},
                                                      {"ab", "cd", "dcba"},
                                                      {"abc", "abd", "cbd"}};
  for (auto test_case : good_test_cases) {
    auto parse_result_or_status =
        ParseEinsumString(to_string(test_case[0], test_case[1], test_case[2]));
    EXPECT_TRUE(parse_result_or_status.status().ok());
    auto parse_result = parse_result_or_status.ValueOrDie();
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(parse_result[i], to_vec(test_case[i]));
    }
    EXPECT_TRUE(ValidateEinsumNumericDimensions(
                    parse_result[0], parse_result[1], parse_result[2])
                    .ok());
  }

  std::vector<string> einsum_strings_that_fail_parsing = {
      "", "a", "ab->ba", "ab,bc,cd->ad", "a...b,bc->a...c"};
  for (auto test_case : einsum_strings_that_fail_parsing) {
    auto parse_result_or_status = ParseEinsumString(test_case);
    EXPECT_FALSE(parse_result_or_status.status().ok());
  }

  std::vector<string> einsum_strings_that_fail_numeric_validation = {
      "a,b->c", "ab,bc->acd", "abz,bc->ac", "ab,bcz->ac"};
  for (auto test_case : einsum_strings_that_fail_numeric_validation) {
    auto parse_result_or_status = ParseEinsumString(test_case);
    EXPECT_TRUE(parse_result_or_status.status().ok());
    auto parse_result = parse_result_or_status.ValueOrDie();
    EXPECT_FALSE(ValidateEinsumNumericDimensions(
                     parse_result[0], parse_result[1], parse_result[2])
                     .ok());
  }
}

}  // namespace
}  // namespace xla
