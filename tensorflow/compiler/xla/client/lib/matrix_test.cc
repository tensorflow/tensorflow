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

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

class MatrixTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  void TestMatrixDiagonal();
  template <typename T>
  void TestMatrixDiagonal4D();
  template <typename T>
  void TestSetMatrixDiagonal();

  template <typename T>
  std::map<int, Array2D<T>> k_and_expected() const {
    return std::map<int, Array2D<T>>{
        {0, {{0, 5, 10}, {12, 17, 22}}},
        {1, {{1, 6, 11}, {13, 18, 23}}},
        {2, {{2, 7}, {14, 19}}},
        {3, {{3}, {15}}},
        {4, {{}, {}}},
        {-1, {{4, 9}, {16, 21}}},
        {-2, {{8}, {20}}},
        {-3, {{}, {}}},
        {-4, {{}, {}}},
    };
  }
};

XLA_TEST_F(MatrixTest, Triangle) {
  XlaBuilder builder(TestName());
  Array3D<int32_t> input(2, 3, 4);
  input.FillIota(0);

  XlaOp a;
  auto a_data = CreateR3Parameter<int32_t>(input, 0, "a", &builder, &a);
  LowerTriangle(a);
  Array3D<int32_t> expected({{{0, 0, 0, 0}, {4, 5, 0, 0}, {8, 9, 10, 0}},
                             {{12, 0, 0, 0}, {16, 17, 0, 0}, {20, 21, 22, 0}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {a_data.get()});
}

XLA_TEST_F(MatrixTest, Symmetrize) {
  for (bool lower : {false, true}) {
    XlaBuilder builder(TestName());
    float nan = std::numeric_limits<float>::quiet_NaN();
    Array<float> input = {
        {1, nan, nan},
        {2, 3, nan},
        {4, 5, 6},
    };

    XlaOp a;
    auto a_data = CreateParameter<float>(input, 0, "a", &builder, &a);
    Symmetrize(lower ? a : TransposeInMinorDims(a), /*lower=*/lower);

    Array<float> expected = {
        {1, 2, 4},
        {2, 3, 5},
        {4, 5, 6},
    };

    ComputeAndCompare<float>(&builder, expected, {a_data.get()});
  }
}

XLA_TEST_F(MatrixTest, SymmetrizeComplex) {
  for (bool lower : {false, true}) {
    XlaBuilder builder(TestName());
    float nan = std::numeric_limits<float>::quiet_NaN();
    Array<complex64> input = {
        {complex64{1, nan}, nan, nan},
        {complex64{2, 7}, complex64{3, nan}, nan},
        {complex64{4, 8}, complex64{5, 9}, complex64{6, nan}},
    };

    XlaOp a;
    auto a_data = CreateParameter<complex64>(input, 0, "a", &builder, &a);
    Symmetrize(lower ? a : Conj(TransposeInMinorDims(a)), /*lower=*/lower);

    Array<complex64> expected = {
        {1, complex64{2, -7}, complex64{4, -8}},
        {complex64{2, 7}, 3, complex64{5, -9}},
        {complex64{4, 8}, complex64{5, 9}, 6},
    };

    ComputeAndCompare<complex64>(&builder, expected, {a_data.get()});
  }
}

template <typename T>
void MatrixTest::TestMatrixDiagonal() {
  XlaBuilder builder("SetMatrixDiagonal");
  Array3D<T> input(2, 3, 4);
  input.FillIota(0);
  for (const auto& kv : k_and_expected<T>()) {
    XlaOp a;
    auto a_data = CreateR3Parameter<T>(input, 0, "a", &builder, &a);
    GetMatrixDiagonal(a, kv.first);

    ComputeAndCompareR2<T>(&builder, kv.second, {a_data.get()});
  }
}

template <typename T>
void MatrixTest::TestSetMatrixDiagonal() {
  XlaBuilder builder("GetMatrixDiagonal");
  Array3D<T> input(2, 3, 4);
  input.FillIota(0);
  for (const auto& kv : k_and_expected<T>()) {
    XlaOp a;
    XlaOp b;
    auto a_data = CreateR3Parameter<T>(input, 0, "a", &builder, &a);
    auto new_diag =
        CreateR2Parameter<T>(Array2D<T>{kv.second}, 1, "d", &builder, &b);

    GetMatrixDiagonal(SetMatrixDiagonal(a, b + ScalarLike(b, 1), kv.first),
                      kv.first) -
        ScalarLike(b, 1);

    ComputeAndCompareR2<T>(&builder, kv.second, {a_data.get(), new_diag.get()});
  }
}

XLA_TEST_F(MatrixTest, SetMatrixDiagonal_S32) {
  TestSetMatrixDiagonal<int32_t>();
}
XLA_TEST_F(MatrixTest, SetMatrixDiagonal_S64) {
  TestSetMatrixDiagonal<int64_t>();
}
XLA_TEST_F(MatrixTest, SetMatrixDiagonal_F32) {
  TestSetMatrixDiagonal<float>();
}

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_S32) { TestMatrixDiagonal<int32_t>(); }

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_S64) { TestMatrixDiagonal<int64_t>(); }

XLA_TEST_F(MatrixTest, GetMatrixDiagonal_F32) { TestMatrixDiagonal<float>(); }

template <typename T>
void MatrixTest::TestMatrixDiagonal4D() {
  XlaBuilder builder("GetMatrixDiagonal");
  Array4D<T> input(2, 2, 4, 3);
  input.FillIota(0);
  std::map<int, Array3D<T>> k_and_expected = {
      {0, {{{0, 4, 8}, {12, 16, 20}}, {{24, 28, 32}, {36, 40, 44}}}},
      {1, {{{1, 5}, {13, 17}}, {{25, 29}, {37, 41}}}},
      {2, {{{2}, {14}}, {{26}, {38}}}},
      {3, {{{}, {}}, {{}, {}}}},
      {4, {{{}, {}}, {{}, {}}}},
      {-1, {{{3, 7, 11}, {15, 19, 23}}, {{27, 31, 35}, {39, 43, 47}}}},
      {-2, {{{6, 10}, {18, 22}}, {{30, 34}, {42, 46}}}},
      {-3, {{{9}, {21}}, {{33}, {45}}}},
      {-4, {{{}, {}}, {{}, {}}}},
  };
  for (const auto& kv : k_and_expected) {
    XlaOp a;
    auto a_data = CreateR4Parameter<T>(input, 0, "a", &builder, &a);
    GetMatrixDiagonal(a, kv.first);

    ComputeAndCompareR3<T>(&builder, kv.second, {a_data.get()});
  }
}

XLA_TEST_F(MatrixTest, GetMatrixDiagonal4D_S32) {
  TestMatrixDiagonal4D<int32_t>();
}

XLA_TEST_F(MatrixTest, GetMatrixDiagonal4D_S64) {
  TestMatrixDiagonal4D<int64_t>();
}

XLA_TEST_F(MatrixTest, GetMatrixDiagonal4D_F32) {
  TestMatrixDiagonal4D<float>();
}

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
      a, {index, ConstantR0<int32_t>(&builder, 0)}, {1, n});
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
      a, {index, ConstantR0<int32_t>(&builder, 0)}, {1, n});
  Einsum(l_index, row, "abc,adc->abd");

  ComputeAndCompareR3<float>(&builder, {{{33}}, {{292}}},
                             {a_data.get(), row_data.get(), index_data.get()});
}

XLA_TEST_F(MatrixTest, ParseEinsumString) {
  auto to_vec = [](absl::string_view s) {
    std::vector<int64_t> v;
    v.reserve(s.size());
    int e = -3;
    for (auto c : s) {
      v.push_back(c == '.' ? e++ : int64_t{c});
    }
    return v;
  };

  auto to_string = [&](absl::string_view x, absl::string_view y,
                       absl::string_view o) {
    return absl::StrCat(x, ",", y, "->", o);
  };

  std::vector<std::vector<std::string>> good_test_cases = {
      {"ab", "bc", "ac"},
      {"Bab", "Bbc", "Bac"},
      {"ab", "cd", "dcba"},
      {"abc", "abd", "cbd"},
      {"...ab", "...bc", "...ac"},
      {"a...bc", "...abd", "cbd..."},
      {"...ab", "...bc", "ac"},
      {"...b", "...bc", "...c"},
      {"...abz", "...bc", "...ac"},
      {"...ab", "...bcz", "...ac"},
      {"abz", "bc", "ac"},
      {"ab", "bcz", "ac"},

      {"a", "b", "c"},
      {"...a", "...b", "...c"},
      {"abb", "bcc", "ac"},
      {"ab", "bc", "ad"},
  };
  for (auto test_case : good_test_cases) {
    auto parse_result_or_status =
        ParseEinsumString(to_string(test_case[0], test_case[1], test_case[2]),
                          test_case[0].size(), test_case[1].size());
    EXPECT_TRUE(parse_result_or_status.status().ok());
    auto parse_result = parse_result_or_status.value();
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(parse_result[i], to_vec(test_case[i]));
    }
  }

  std::vector<std::string> einsum_strings_that_fail_parsing = {
      "", "a", "ab->ba", "ab,bc,cd->ad", "a...b...,bc->a...c",
  };
  for (auto test_case : einsum_strings_that_fail_parsing) {
    auto parse_result_or_status = ParseEinsumString(test_case, 3, 3);
    EXPECT_FALSE(parse_result_or_status.status().ok());
  }
}

XLA_TEST_F(MatrixTest, NormalizeEinsumString) {
  EXPECT_EQ(NormalizeEinsumString("a,b->ab"), "");
  EXPECT_EQ(NormalizeEinsumString("ba"), "ba->ab");
  EXPECT_EQ(NormalizeEinsumString("ab,dc"), "ab,dc->abcd");
  EXPECT_EQ(NormalizeEinsumString("a,b"), "a,b->ab");
  EXPECT_EQ(NormalizeEinsumString("...ba,ca..."), "...ba,ca...->...bc");
}

}  // namespace
}  // namespace xla
