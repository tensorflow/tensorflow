/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/reference_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"

namespace xla {
namespace {

constexpr ErrorSpec kErrorSpec{0.0001};

class TransposeTest : public ClientLibraryTestRunnerMixin<
                          HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 protected:
  void TestTransposeConstant(Vector3 sizes, Vector3 transpose_dims) {
    Array3D<int32_t> aoperand(sizes[0], sizes[1], sizes[2]);
    std::vector<int32_t> expected(sizes[0] * sizes[1] * sizes[2]);
    for (int64_t i = 0; i < sizes[0]; ++i) {
      for (int64_t j = 0; j < sizes[1]; ++j) {
        for (int64_t k = 0; k < sizes[2]; ++k) {
          Vector3 indices{i, j, k};
          aoperand(i, j, k) = (i * sizes[1] + j) * sizes[2] + k;
          expected[(indices[transpose_dims[0]] * sizes[transpose_dims[1]] +
                    indices[transpose_dims[1]]) *
                       sizes[transpose_dims[2]] +
                   indices[transpose_dims[2]]] = aoperand(i, j, k);
        }
      }
    }

    XlaBuilder builder(TestName());
    auto operand = ConstantR3FromArray3D(&builder, aoperand);
    auto transpose = Transpose(operand, transpose_dims);
    // Add a reshape so that the transpose does not disappear during layout
    // assignment.
    Reshape(transpose, {sizes[0] * sizes[1] * sizes[2]});

    ComputeAndCompareR1<int32_t>(&builder, expected, {});
  }
};

TEST_F(TransposeTest, Transpose0x0) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 0));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {}, kErrorSpec);
}

TEST_F(TransposeTest, Transpose0x42) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 42));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(42, 0), {}, kErrorSpec);
}

TEST_F(TransposeTest, Transpose7x0) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(7, 0));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 7), {}, kErrorSpec);
}

TEST_F(TransposeTest, Transpose2x2) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2<float>(&builder, {
                                             {1.0, 2.0},
                                             {3.0, 4.0},
                                         });
  Transpose(lhs, {1, 0});

  Array2D<float> expected({{1.0f, 3.0f}, {2.0f, 4.0f}});

  ComputeAndCompareR2<float>(&builder, expected, {}, kErrorSpec);
}

TEST_F(TransposeTest, Transpose0x2x3_2x3x0) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, Array3D<int32_t>(0, 2, 3));
  Transpose(operand, {1, 2, 0});

  ComputeAndCompareR3<int32_t>(&builder, Array3D<int32_t>(2, 3, 0), {});
}

TEST_F(TransposeTest, Transpose1x2x3_2x3x1) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {1, 2, 0});

  Array3D<int32_t> expected({{{1}, {2}, {3}}, {{4}, {5}, {6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_3x2x1) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {2, 1, 0});

  Array3D<int32_t> expected({{{1}, {4}}, {{2}, {5}}, {{3}, {6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_1x2x3) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {0, 1, 2});

  Array3D<int32_t> expected({{{1, 2, 3}, {4, 5, 6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, MultiTranspose3x2) {
  Array2D<float> input({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  Array2D<float> transposed({{1.0f, 3.0f, 5.0f}, {2.0f, 4.0f, 6.0f}});

  for (int transposes = 0; transposes <= 10; ++transposes) {
    XlaBuilder builder("Transpose");
    auto computed = ConstantR2FromArray2D<float>(&builder, input);
    for (int i = 0; i < transposes; ++i) {
      computed = Transpose(computed, {1, 0});
    }
    const Array2D<float>& expected = transposes % 2 == 0 ? input : transposed;
    ComputeAndCompareR2<float>(&builder, expected, {}, kErrorSpec);
  }
}

// Test for transposing [1x1] matrix.
TEST_F(TransposeTest, Small_1x1) {
  auto aoperand = MakeLinspaceArray2D(0.0, 1.0, 1, 1);

  XlaBuilder builder("transpose_1x1");
  auto operand = ConstantR2FromArray2D<float>(&builder, *aoperand);
  Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

// Test for transposing [2x2] matrix.
TEST_F(TransposeTest, Small_2x2) {
  auto aoperand = MakeLinspaceArray2D(0.0, 4.0, 2, 2);

  XlaBuilder builder("transpose_2x2");
  auto operand = ConstantR2FromArray2D<float>(&builder, *aoperand);
  Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

TEST_F(TransposeTest, TransposeConstant021_SingleIncompleteTilePerLayer) {
  TestTransposeConstant({2, 16, 17}, {0, 2, 1});
}

TEST_F(TransposeTest, TransposeConstant021_SingleCompleteTilePerLayer) {
  TestTransposeConstant({2, 32, 32}, {0, 2, 1});
}

TEST_F(TransposeTest, TransposeConstant021_MultipleTilesPerLayer) {
  TestTransposeConstant({2, 70, 35}, {0, 2, 1});
}

TEST_F(TransposeTest, TransposeConstant210_DegenerateDim) {
  TestTransposeConstant({20, 30, 1}, {2, 1, 0});
}

using HloTransposeTest = HloPjRtTestBase;

// Disable HLO passes to verify the default behavior
TEST_F(HloTransposeTest, HloPassesDisabled) {
  if (test::DeviceTypeIsOneOf({test::kGpu, test::kInterpreter, test::kTpu})) {
    GTEST_SKIP();
  }
  const char* const kModuleStr = R"(
    HloModule Transpose

    ENTRY Transpose {
      constant = s32[2,3] constant({ { 1, 2, 3 }, { 4, 5, 6 } })
      ROOT transpose = s32[3,2] transpose(constant), dimensions={1,0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(std::move(module), {}, /*run_hlo_passes=*/false));
  Array2D<int32_t> array({{1, 4}, {2, 5}, {3, 6}});
  auto expected = LiteralUtil::CreateR2FromArray2D(array);

  EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace xla
