/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class TransposeTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};

 protected:
  void TestTransposeConstant021(size_t n1, size_t n2, size_t n3);
};

XLA_TEST_F(TransposeTest, Transpose0x0) {
  ComputationBuilder builder(client_, "Transpose");
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 0));
  auto result = builder.Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose0x42) {
  ComputationBuilder builder(client_, "Transpose");
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 42));
  auto result = builder.Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(42, 0), {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose7x0) {
  ComputationBuilder builder(client_, "Transpose");
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(7, 0));
  auto result = builder.Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 7), {}, error_spec_);
}

TEST_F(TransposeTest, Transpose2x2) {
  ComputationBuilder builder(client_, "Transpose");
  auto lhs = builder.ConstantR2<float>({
      {1.0, 2.0}, {3.0, 4.0},
  });
  auto result = builder.Transpose(lhs, {1, 0});

  Array2D<float> expected({{1.0f, 3.0f}, {2.0f, 4.0f}});

  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose0x2x3_2x3x0) {
  ComputationBuilder builder(client_, "Transpose");
  auto operand = builder.ConstantR3FromArray3D<int32>(Array3D<int32>(0, 2, 3));
  auto result = builder.Transpose(operand, {1, 2, 0});

  ComputeAndCompareR3<int32>(&builder, Array3D<int32>(2, 3, 0), {});
}

TEST_F(TransposeTest, Transpose1x2x3_2x3x1) {
  ComputationBuilder builder(client_, "Transpose");
  auto operand = builder.ConstantR3FromArray3D<int32>({{{1, 2, 3}, {4, 5, 6}}});
  auto result = builder.Transpose(operand, {1, 2, 0});

  Array3D<int32> expected({{{1}, {2}, {3}}, {{4}, {5}, {6}}});

  ComputeAndCompareR3<int32>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_3x2x1) {
  ComputationBuilder builder(client_, "Transpose");
  auto operand = builder.ConstantR3FromArray3D<int32>({{{1, 2, 3}, {4, 5, 6}}});
  auto result = builder.Transpose(operand, {2, 1, 0});

  Array3D<int32> expected({{{1}, {4}}, {{2}, {5}}, {{3}, {6}}});

  ComputeAndCompareR3<int32>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_1x2x3) {
  ComputationBuilder builder(client_, "Transpose");
  auto operand = builder.ConstantR3FromArray3D<int32>({{{1, 2, 3}, {4, 5, 6}}});
  auto result = builder.Transpose(operand, {0, 1, 2});

  Array3D<int32> expected({{{1, 2, 3}, {4, 5, 6}}});

  ComputeAndCompareR3<int32>(&builder, expected, {});
}

TEST_F(TransposeTest, MultiTranspose3x2) {
  Array2D<float> input({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  Array2D<float> transposed({{1.0f, 3.0f, 5.0f}, {2.0f, 4.0f, 6.0f}});

  for (int transposes = 0; transposes <= 10; ++transposes) {
    ComputationBuilder builder(client_, "Transpose");
    auto computed = builder.ConstantR2FromArray2D<float>(input);
    for (int i = 0; i < transposes; ++i) {
      computed = builder.Transpose(computed, {1, 0});
    }
    const Array2D<float>& expected = transposes % 2 == 0 ? input : transposed;
    ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
  }
}

// Test for transposing [1x1] matrix.
TEST_F(TransposeTest, Small_1x1) {
  auto aoperand = MakeLinspaceArray2D(0.0, 1.0, 1, 1);

  ComputationBuilder builder(client_, "transpose_1x1");
  auto operand = builder.ConstantR2FromArray2D<float>(*aoperand);
  builder.Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

// Test for transposing [2x2] matrix.
TEST_F(TransposeTest, Small_2x2) {
  auto aoperand = MakeLinspaceArray2D(0.0, 4.0, 2, 2);

  ComputationBuilder builder(client_, "transpose_2x2");
  auto operand = builder.ConstantR2FromArray2D<float>(*aoperand);
  builder.Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

void TransposeTest::TestTransposeConstant021(size_t n1, size_t n2, size_t n3) {
  Array3D<int32> aoperand(n1, n2, n3);
  Array3D<int32> expected(n1, n3, n2);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        aoperand(i, j, k) = i * n3 * n2 + j * n3 + k;
        expected(i, k, j) = aoperand(i, j, k);
      }
    }
  }

  ComputationBuilder builder(client_, TestName());
  auto operand = builder.ConstantR3FromArray3D(aoperand);
  builder.Transpose(operand, {0, 2, 1});

  ComputeAndCompareR3<int32>(&builder, expected, {});
}

TEST_F(TransposeTest, TransposeConstant021_SingleIncompleteTilePerLayer) {
  TestTransposeConstant021(2, 2, 3);
}

TEST_F(TransposeTest, TransposeConstant021_SingleCompleteTilePerLayer) {
  TestTransposeConstant021(2, 32, 32);
}

TEST_F(TransposeTest, TransposeConstant021_MultipleTilesPerLayer) {
  TestTransposeConstant021(2, 70, 35);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
