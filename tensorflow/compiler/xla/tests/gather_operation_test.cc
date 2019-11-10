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

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using absl::nullopt;

class GatherOperationTest : public HloTestBase {
 protected:
  void RunTest(const string& hlo_text, Literal* operand,
               Literal* start_indices) {
    RunTest(hlo_text, {operand, start_indices});
  }

  void RunTest(const string& hlo_text, absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, nullopt));
  }
};

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV1) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV2) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherMultipleBatchDims) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_0) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_1) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNd) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({1, 1});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, BatchDynamicSlice) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
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
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, OutOfBoundsIndex) {
  // Out of bounds indices must not crash, and the indices in range should
  // produce the same values across all backends.

  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, OutOfBoundsUnsignedIndex) {
  // Out of bounds indices must not crash, and the indices in range should
  // produce the same values across all backends.

  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<uint32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483648u, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, NegativeIndex) {
  // Negative indices must not crash, and the indices in range should produce
  // the same values across all backends.

  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>(
      {{2, -1}, {2, 1}, {1, 1}, {-500, 1}, {-2147483648, 1}, {1, 2}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, NegativeIndexIntoUnsignedOperand) {
  // Negative indices must not crash, and the indices in range should produce
  // the same values across all backends.

  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<uint32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>(
      {{2, -1}, {2, 1}, {1, 1}, {-500, 1}, {-2147483648, 1}, {1, 2}});
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
  Literal operand = LiteralUtil::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  Literal start_indices = LiteralUtil::CreateR0<int32>(1);
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
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3, 4});
  Literal start_indices = LiteralUtil::CreateR0<int32>(1);
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, ZeroSizedResult) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherV2) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherMultipleBatchDims) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherNdMultipleBatchDims) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices =
      LiteralUtil::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedTensorFlowGatherNd) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest,
           FusedTensorFlowGatherNdNonDefaultIndexVectorDim) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({1, 1});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, FusedBatchDynamicSlice) {
  const string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  RunTest(hlo_text, &operand, &start_indices);
}

XLA_TEST_F(GatherOperationTest, GatherFromScalar) {
  const string hlo_text = R"(
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
  Literal start_indices = LiteralUtil::CreateR1<int32>({});
  RunTest(hlo_text, &operand, &start_indices);
}

class GatherClientLibraryTest : public ClientLibraryTestBase {};

// Disabled on interpreter since ExectuteAsyncOnStream is not supported.
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

  std::vector<int32> expected = {};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> operand_arg,
      client_->TransferToServer(
          LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> indices_arg,
      client_->TransferToServer(LiteralUtil::CreateR1<int32>({0, 2})));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::DeviceHandle> devices,
                          client_->GetDeviceHandles(1));
  xla::ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  *execution_options.add_device_handles() = devices[0];
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  std::vector<xla::Client::XlaComputationInstance> computation_instances = {
      {computation,
       {operand_arg.get(), indices_arg.get()},
       execution_options,
       /*execution_profile=*/nullptr}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<xla::GlobalData>> result_data,
      client_->ExecuteParallel(computation_instances));
  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          client_->Transfer(*(result_data[0])));
  LiteralTestUtil::ExpectR2Equal<int32>({{1, 2, 3}, {7, 8, 9}}, result_literal);
}
}  // namespace
}  // namespace xla
