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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/error_spec.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class GatherOperationTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 protected:
  void RunTest(const std::string& hlo_text, Literal* operand,
               Literal* start_indices) {
    RunTest(hlo_text, {operand, start_indices});
  }

  void RunTest(const std::string& hlo_text, absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, std::nullopt));
  }
};

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV1) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, BatchDimInMiddle) {
  // Reverse the middle dimension (dim 1).
  const std::string hlo_text = R"(
HloModule BatchDimInMiddle

ENTRY main {
  operand = s32[3, 2, 3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3, 1, 2, 3] gather(operand, indices),
      offset_dims={0, 1, 3},
      collapsed_slice_dims={},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1, 3}
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{1, 2, 3}, {4, 5, 6}},
                                      {{7, 8, 9}, {10, 11, 12}},
                                      {{13, 14, 15}, {16, 17, 18}}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({1, 0});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV2) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherMultipleBatchDims) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_0) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=2,
      slice_sizes={1, 1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32_t>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_1) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,1,1,2] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=2,
      slice_sizes={1, 1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32_t>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNd) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, BatchDynamicSlice) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 0}
}
)";
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, OutOfBoundsIndex) {
  // Out of bounds indices must not crash, and the indices in range should
  // produce the same values across all backends.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = s32[6]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

// The next 2 tests uses data types that require extra steps on some backends so
// only run them on known good backends.
#if defined(XLA_TEST_BACKEND_GPU) || defined(XLA_TEST_BACKEND_CPU) || \
    defined(XLA_TEST_BACKEND_INTERPRETER)

XLA_TEST_F(GatherOperationTest, OutOfBoundsIndex64Bit) {
  // Out of bounds indices must not crash, even when the value is of a type
  // larger than needed to access all values in the input, and the indices
  // produce the same values across all backends.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s64[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = s32[6]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int64_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {21474836407, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TooSmallIndex8Bit) {
  // Indices of a type too small to index all locations in gather should not
  // fail.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[512, 512]{1,0} parameter(0)
  indices = u8[7,2]{1,0} parameter(1)
  gather = s32[7,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = s32[7]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateRandomLiteral<S32>(
          ShapeUtil::MakeShape(S32, {512, 512}), /*mean=*/1000, /*stddev=*/500)
          .value();
  Literal start_indices = LiteralUtil::CreateR2<uint8_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {7, 1}, {1, 2}, {0x80, 0x80}});
  RunTest(hlo_text, &operand, &start_indices);
}

#endif

XLA_TEST_F(GatherOperationTest, OutOfBoundsUnsignedIndex) {
  // Out of bounds indices must not crash, and the indices in range should
  // produce the same values across all backends.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = u32[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = s32[6]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<uint32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483648u, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, NegativeIndex) {
  // Negative indices must not crash, and the indices in range should produce
  // the same values across all backends.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  gather = s32[6,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = s32[6]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR2<int32_t>({{2, -1},
                                      {2, 1},
                                      {1, 1},
                                      {-500, 1},
                                      {static_cast<int32_t>(-2147483648), 1},
                                      {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, NegativeIndexIntoUnsignedOperand) {
  // Negative indices must not crash, and the indices in range should produce
  // the same values across all backends.

  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = u32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  gather = u32[6,1,1]{2,1,0} gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
  ROOT result = u32[6]{0} reshape(gather)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR2<int32_t>({{2, -1},
                                      {2, 1},
                                      {1, 1},
                                      {-500, 1},
                                      {static_cast<int32_t>(-2147483648), 1},
                                      {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, OneScalarIndex) {
  const char* hlo_text = R"(
HloModule OneScalarIndex

ENTRY main {
  operand = s32[2,3,2]{2,1,0} parameter(0)
  index = s32[] parameter(1)
  ROOT gather = s32[1,3,2]{2,1,0} gather(operand, index),
      offset_dims={0,1,2},
      collapsed_slice_dims={},
      start_index_map={0},
      index_vector_dim=0,
      slice_sizes={1,3,2}
}
)";
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  Literal start_indices = LiteralUtil::CreateR0<int32_t>(1);
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, ScalarResult) {
  const char* hlo_text = R"(
HloModule ScalarResult

ENTRY main {
  operand = s32[4]{0} parameter(0)
  index = s32[] parameter(1)
  ROOT gather = s32[] gather(operand, index),
      offset_dims={},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=0,
      slice_sizes={1}
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal start_indices = LiteralUtil::CreateR0<int32_t>(1);
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, ZeroSizedResult) {
  const std::string hlo_text = R"(
HloModule ZeroSizedResult

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[0] parameter(1)
  ROOT gather = s32[0,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherV2) {
  const std::string hlo_text = R"(
HloModule FusedTensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[3,2] broadcast(one), dimensions={}
  ROOT result = s32[3,2]{1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherMultipleBatchDims) {
  const std::string hlo_text = R"(
HloModule FusedTensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,3,2] broadcast(one), dimensions={}
  ROOT result = s32[2,3,2]{2,1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherNdMultipleBatchDims) {
  const std::string hlo_text = R"(
HloModule FusedTensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=2,
      slice_sizes={1, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32_t>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherNd) {
  const std::string hlo_text = R"(
HloModule FusedTensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest,
           FusedTensorFlowGatherNdNonDefaultIndexVectorDim) {
  const std::string hlo_text = R"(
HloModule FusedTensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedDynamicSlice) {
  const char* hlo_text = R"(
HloModule FusedDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
  one = s32[] constant(1)
  one_broadcasted = s32[1,1] broadcast(one), dimensions={}
  ROOT result = s32[1,1]{1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedBatchDynamicSlice) {
  const std::string hlo_text = R"(
HloModule FusedBatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,1,1] broadcast(one), dimensions={}
  ROOT result = s32[2,1,1]{2,1,0} add(gather, one_broadcasted)
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, GatherFromScalar) {
  const std::string hlo_text = R"(
HloModule GatherFromScalar

ENTRY main {
  operand = f32[] parameter(0)
  indices = s32[0]{0} parameter(1)
  ROOT gather = f32[] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={},
      start_index_map={},
      index_vector_dim=0,
      slice_sizes={}
}
)";
  Literal operand = LiteralUtil::CreateR0<float>(1);
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, GatherFromScalarNonZeroIndices) {
  const std::string hlo_text = R"(
HloModule GatherFromScalar

ENTRY main {
  operand = f32[1,1,1] parameter(0)
  indices = s32[2,3,50] parameter(1)
  ROOT gather = f32[1,2,50] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={0,1},
      start_index_map={1,0,2},
      index_vector_dim=1,
      slice_sizes={1,1,1}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0, 0}));
}

class GatherClientLibraryTest : public ClientLibraryTestBase {};

// Disabled on interpreter since ExecuteAsyncOnStream is not supported.
XLA_TEST_F(GatherClientLibraryTest,
           DISABLED_ON_INTERPRETER(DISABLED_ON_GPU(Basic))) {
  // We create this HLO, but using the XlaBuilder API.
  //
  // ENTRY main {
  //   operand = s32[3,3] parameter(0)
  //   indices = s32[2] parameter(1)
  //   ROOT gather = s32[2,3] gather(operand, indices),
  //       offset_dims={1},
  //       collapsed_slice_dims={0},
  //       start_index_map={0},
  //       index_vector_dim=1,
  //       slice_sizes={1, 3}
  // }

  XlaBuilder builder("gather_basic");

  Shape operand_shape = ShapeUtil::MakeShape(S32, {3, 3});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});

  auto operand = Parameter(&builder, 0, operand_shape, "operand");
  auto indices = Parameter(&builder, 1, indices_shape, "indices");
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_collapsed_slice_dims(0);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  Gather(operand, indices, dim_numbers, {1, 3});

  std::vector<int32_t> expected = {};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> operand_arg,
      client_->TransferToServer(
          LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> indices_arg,
      client_->TransferToServer(LiteralUtil::CreateR1<int32_t>({0, 2})));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::DeviceHandle> devices,
                          client_->GetDeviceHandles(1));
  xla::ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  *execution_options.add_device_handles() = devices[0];
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  std::vector<xla::XlaComputationInstance> computation_instances = {
      {computation,
       {operand_arg.get(), indices_arg.get()},
       execution_options,
       /*execution_profile=*/nullptr}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<xla::GlobalData>> result_data,
      client_->ExecuteParallel(computation_instances));
  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          client_->Transfer(*(result_data[0])));
  LiteralTestUtil::ExpectR2Equal<int32_t>({{1, 2, 3}, {7, 8, 9}},
                                          result_literal);
}

XLA_TEST_F(GatherOperationTest, b_301618442_case1) {
  const std::string hlo_text = R"(
HloModule b_301618442

ENTRY main {
  operand = s32[4,256] parameter(0)
  indices = s32[64] parameter(1)
  ROOT gather = s32[64,256] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,256}
}
)";
  Array<int32_t> operand({4, 256});
  operand.FillIota(0);
  Literal operand_literal = LiteralUtil::CreateFromArray(operand);
  Array<int32_t> indices({64});
  operand.FillRandomUniform(0, 3);
  Literal indices_literal = LiteralUtil::CreateFromArray(indices);
  RunTest(hlo_text, &operand_literal, &indices_literal);
}

XLA_TEST_F(GatherOperationTest, b_301618442_case2) {
  const std::string hlo_text = R"(
HloModule b_301618442

ENTRY main {
  operand = s32[4,4096] parameter(0)
  indices = s32[64] parameter(1)
  ROOT gather = s32[64,4096] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4096}
}
)";
  Array<int32_t> operand({4, 4096});
  operand.FillIota(0);
  Literal operand_literal = LiteralUtil::CreateFromArray(operand);
  Array<int32_t> indices({64});
  operand.FillRandomUniform(0, 3);
  Literal indices_literal = LiteralUtil::CreateFromArray(indices);
  RunTest(hlo_text, &operand_literal, &indices_literal);
}

}  // namespace
}  // namespace xla
