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

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using absl::nullopt;

class ScatterTest : public HloTestBase {
 protected:
  void RunTest(const string& hlo_text, Literal* operand,
               Literal* scatter_indices, Literal* updates) {
    RunTest(hlo_text, {operand, scatter_indices, updates});
  }

  void RunTest(const string& hlo_text, absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseHloString(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, nullopt));
  }
};

XLA_TEST_F(ScatterTest, TensorFlowScatterV1_Update) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterV1_WithFusedAdds) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  p0 = s32[3,3] parameter(0)
  operand = s32[3,3] add(p0, p0)
  p1 = s32[2] parameter(1)
  indices = s32[2] add(p1, p1)
  p2 = s32[2,3] parameter(2)
  updates = s32[2,3] add(p2, p2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterV2_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32>({{10, 30}, {40, 60}, {70, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterV2_InversePermutation) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  permutation = s32[3,4] parameter(0)
  reshape = s32[3,4,1] reshape(permutation)
  operand = s32[3,4] iota(), iota_dimension=1
  updates = s32[3,4,1,1] iota(), iota_dimension=1
  iota = s32[3,4,1] iota(), iota_dimension=0
  indices = s32[3,4,2] concatenate(iota, reshape), dimensions={2}
  ROOT scatter = s32[3,4] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=2
}
)";
  Literal permutation =
      LiteralUtil::CreateR2<int32>({{1, 3, 2, 0}, {3, 0, 2, 1}, {2, 3, 1, 0}});
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_text, config));
  auto actual = ExecuteAndTransfer(std::move(module), {&permutation});
  Literal expected =
      LiteralUtil::CreateR2<int32>({{3, 0, 2, 1}, {1, 3, 2, 0}, {3, 2, 0, 1}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

XLA_TEST_F(ScatterTest, SimpleR4) {
  const char* hlo_text = R"(
HloModule SimpleR4

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[1,2,2,1] parameter(0)
  indices = s32[1,3] parameter(1)
  updates = f32[1,2,2,1] parameter(2)
  ROOT scatter = f32[1,2,2,1] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1,2,3},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0, 2, 1},
      index_vector_dim=1
}
)";

  Literal operand =
      LiteralUtil::CreateR4<float>({{{{0.f}, {0.f}}, {{0.f}, {0.f}}}});
  Literal updates =
      LiteralUtil::CreateR4<float>({{{{0.12}, {0.28}}, {{0.018}, {0.42}}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0, 0}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Add) {
  const string hlo_text = R"(
HloModule TensorFlowScatter_Add

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Mul) {
  const string hlo_text = R"(
HloModule TensorFlowScatter_Mul

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=mul_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_F32) {
  const string hlo_text = R"(
HloModule TensorFlowScatter_F32

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_RepeatedIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_MultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterMultipleBatchDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=2
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterNd) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-40, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterNd_NonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNdNonDefaultIndexVectorDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-20, 20}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, DynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule DynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0,1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, BatchDynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{10}}, {{20}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_ZeroDimBounds

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,0] parameter(2)
  ROOT scatter = s32[3,0] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{}, {}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, NoUpdateWindowDims) {
  const string hlo_text = R"(
HloModule Scatter_NoUpdateWindowDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2
}
)";
  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20}, {30, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OutOfBoundsIndex) {
  const string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OutOfBoundsUnsignedIndex) {
  const string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = u32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<uint32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483648u, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, NegativeIndex) {
  const string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {-500, 1}, {-2147483648, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OutOfBoundsUpdateWindow) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd_OobUpdateWindow

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[1,2] parameter(1)
  updates = s32[1,2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{-10, 10}, {-40, 40}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OneScalarIndex) {
  const char* hlo_text = R"(
HloModule OneScalarIndex

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[2,3,2]{2,1,0} parameter(0)
  index = s32[] parameter(1)
  updates = s32[1,3,2]{2,1,0} parameter(2)
  ROOT scatter = s32[2,3,2]{2,1,0} scatter(operand, index, updates),
      to_apply=update_s32,
      update_window_dims={0,1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0
}
)";
  Literal operand = LiteralUtil::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  Literal scatter_indices = LiteralUtil::CreateR0<int32>(1);
  Literal updates =
      LiteralUtil::CreateR3<int32>({{{10, 20}, {30, 40}, {50, 60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, ScalarUpdate) {
  const char* hlo_text = R"(
HloModule ScalarUpdate

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[4]{0} parameter(0)
  index = s32[] parameter(1)
  updates = s32[] parameter(2)
  ROOT scatter = s32[4]{0} scatter(operand, index, updates),
      to_apply=update_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0
}
)";
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3, 4});
  Literal scatter_indices = LiteralUtil::CreateR0<int32>(1);
  Literal updates = LiteralUtil::CreateR0<int32>(25);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, EmptyIndices) {
  const string hlo_text = R"(
HloModule EmptyIndices

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[0] parameter(1)
  updates = s32[0] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({});
  Literal updates = LiteralUtil::CreateR1<int32>({});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, ScatterIntoScalar) {
  const char* hlo_text = R"(
HloModule ScatterIntoScalar

update_s32 {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  parameter.1 = s32[] parameter(0)
  parameter.2 = s32[0]{0} parameter(1)
  parameter.3 = s32[] parameter(2)
  ROOT scatter = s32[] scatter(parameter.1, parameter.2, parameter.3),
      update_window_dims={},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=0,
      to_apply=update_s32
}
)";
  Literal operand = LiteralUtil::CreateR0<int32>(1);
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({});
  Literal updates = LiteralUtil::CreateR0<int32>(2);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

}  // namespace
}  // namespace xla
