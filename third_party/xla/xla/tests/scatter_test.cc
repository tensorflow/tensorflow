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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla {
namespace {

class ScatterTest : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 protected:
  void RunTest(const absl::string_view hlo_text, Literal* const operand,
               Literal* const scatter_indices, Literal* const updates) {
    RunTest(hlo_text, {operand, scatter_indices, updates});
  }

  void RunTest(const absl::string_view hlo_text,
               const absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, std::nullopt));
  }
};

XLA_TEST_F(ScatterTest, TensorFlowScatterV1_Update) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterV1_WithFusedAdds) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 30}, {40, 60}, {70, 90}});
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
  Literal permutation = LiteralUtil::CreateR2<int32_t>(
      {{1, 3, 2, 0}, {3, 0, 2, 1}, {2, 3, 1, 0}});
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  auto actual = ExecuteAndTransfer(std::move(module), {&permutation});
  Literal expected = LiteralUtil::CreateR2<int32_t>(
      {{3, 0, 2, 1}, {1, 3, 2, 0}, {3, 2, 0, 1}});
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
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0, 0}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Add) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Add_UniqueIndices) {
  const std::string hlo_text = R"(
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
      index_vector_dim=1,
      unique_indices=true
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Mul) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_F32) {
  const std::string hlo_text = R"(
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
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_F16) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_F16

add_f16 (lhs: f16[], rhs: f16[]) -> f16[] {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT add = f16[] add(f16[] lhs, f16[] rhs)
}

ENTRY main {
  operand = f16[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f16[2,3] parameter(2)
  ROOT scatter = f16[3,3] scatter(operand, indices, updates),
      to_apply=add_f16,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Array2D<Eigen::half> operand_array(
      {{1.1f, 2.2f, 3.3f}, {4.4f, 5.5f, 6.6f}, {7.7f, 8.8f, 9.9f}});
  Literal operand(ShapeUtil::MakeShape(F16, {3, 3}));
  operand.PopulateR2FromArray2D(operand_array);
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Array2D<Eigen::half> updates_array({{0.4f, 1.1f, 0.7f}, {2.3f, 3.1f, 1.6f}});
  Literal updates(ShapeUtil::MakeShape(F16, {2, 3}));
  updates.PopulateR2FromArray2D(updates_array);
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, TensorFlowScatterNdS64) {
  constexpr char hlo_text[] = R"(
HloModule S64Scatter

update {
  lhs = s64[] parameter(0)
  ROOT rhs = s64[] parameter(1)
}

ENTRY main {
  operand = s64[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s64[2,2] parameter(2)
  ROOT scatter = s64[3,3,2] scatter(operand, indices, updates),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int64_t>({{{-1, 1LL << 62}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6LL << 59}},  //
                                      {{-7, 7}, {-8, 8LL << 49}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates =
      LiteralUtil::CreateR2<int64_t>({{-10, 10LL << 46}, {-(4LL << 38), 40}});
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-20, 20}});
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10}});
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{10}}, {{20}}});
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
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{}, {}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, NoUpdateWindowDims) {
  const std::string hlo_text = R"(
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
  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32_t>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10, 20}, {30, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OutOfBoundsIndex) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, OutOfBoundsUnsignedIndex) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<uint32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483648u, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, U8Index) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[129,3]{1,0} parameter(0)
  indices = u8[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[129,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateRandomLiteral<S32>(ShapeUtil::MakeShape(S32, {129, 3}),
                                            /*mean=*/500, /*stddev=*/100)
          .value();
  Literal scatter_indices = LiteralUtil::CreateR2<uint8_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {0x80, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, NegativeIndex) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices =
      LiteralUtil::CreateR2<int32_t>({{2, 7},
                                      {2, 1},
                                      {1, 1},
                                      {-500, 1},
                                      {static_cast<int32_t>(-2147483648), 1},
                                      {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{-10, 10}, {-40, 40}}});
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
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  Literal scatter_indices = LiteralUtil::CreateR0<int32_t>(1);
  Literal updates =
      LiteralUtil::CreateR3<int32_t>({{{10, 20}, {30, 40}, {50, 60}}});
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
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal scatter_indices = LiteralUtil::CreateR0<int32_t>(1);
  Literal updates = LiteralUtil::CreateR0<int32_t>(25);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

XLA_TEST_F(ScatterTest, EmptyIndices) {
  const std::string hlo_text = R"(
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
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({});
  Literal updates = LiteralUtil::CreateR1<int32_t>({});
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
  Literal operand = LiteralUtil::CreateR0<int32_t>(1);
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({});
  Literal updates = LiteralUtil::CreateR0<int32_t>(2);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

// TODO(b/230137437): Enable this on GPU once mhlo allows variadic scatter.
XLA_TEST_F(ScatterTest, DISABLED_ON_GPU(Multioutput)) {
  constexpr char hlo_text[] = R"(
HloModule MultioutputScatter

update {
  lhs0 = s32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = s32[] parameter(2)
  rhs1 = f32[] parameter(3)
  ROOT tuple = (s32[], f32[]) tuple(rhs0, rhs1)
}

ENTRY main {
  operand0 = s32[3,3,2] parameter(0)
  operand1 = f32[3,3,2] parameter(1)
  indices = s32[2,2] parameter(2)
  updates0 = s32[2,2] parameter(3)
  updates1 = f32[2,2] parameter(4)
  ROOT scatter = (s32[3,3,2], f32[3,3,2]) scatter(operand0, operand1, indices, updates0, updates1),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand0 =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal operand1 =
      LiteralUtil::CreateR3<float>({{{-2, 2}, {-3, 3}, {-4, 4}},  //
                                    {{-5, 5}, {-6, 6}, {-7, 7}},  //
                                    {{-8, 8}, {-9, 9}, {-10, 10}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates0 = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  Literal updates1 = LiteralUtil::CreateR2<float>({{-11, 11}, {-41, 41}});
  RunTest(hlo_text,
          {&operand0, &operand1, &scatter_indices, &updates0, &updates1});
}

XLA_TEST_F(ScatterTest, TensorFlowScatter_Max_F32) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_Max_F32

max_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=max_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

// Test min/max/add scatters with edge-case values.
class ScatterEdgeCaseTestP
    : public ScatterTest,
      public ::testing::WithParamInterface<absl::string_view /*operator*/> {};

XLA_TEST_P(ScatterEdgeCaseTestP, DoIt) {
  using L = std::numeric_limits<float>;
  std::vector<float> edge_cases = {
      0.f,
      -0.f,
      -1.f,
      1.f,
      L::min(),
      -L::min(),
      L::max(),
      L::lowest(),
      L::epsilon(),
      L::infinity(),
      -L::infinity(),
      L::quiet_NaN(),
      -L::quiet_NaN(),
  };
  int n = edge_cases.size();

  float init_value;
  absl::string_view operation = GetParam();
  if (operation == "minimum") {
    init_value = L::infinity();
  } else if (operation == "maximum") {
    init_value = -L::infinity();
  } else if (operation == "add") {
    init_value = 0;
  } else {
    FAIL() << "Invalid operation " << operation;
  }

  // For each pair of values (x,y) in edge_cases, let the initial value be
  // init_value.  Scatter x and y into the same location.
  const std::string hlo_text = absl::Substitute(R"(
HloModule test

max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = maximum(lhs, rhs)
}

ENTRY main {
  init = f32[$0, $0] broadcast(f32[] constant($2))
  indices = s32[$1] parameter(0)
  updates = f32[$1, $0] parameter(1)
  ROOT scatter = f32[$0, $0] scatter(init, indices, updates),
      to_apply=max_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)",
                                                n, 2 * n, init_value);

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_enable_fast_min_max(false);

  HloModuleConfig config;
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  Literal scatter_indices(ShapeUtil::MakeShape(S32, {2 * n}));
  Literal updates(ShapeUtil::MakeShape(F32, {2 * n, n}));
  for (int i = 0; i < n; i++) {
    scatter_indices.Set({i}, i);
    scatter_indices.Set({i + n}, i);
    for (int j = 0; j < n; j++) {
      updates.Set({i, j}, edge_cases[i]);
      updates.Set({i + n, j}, edge_cases[j]);
    }
  }

  // Expect exact numerical matches.
  ErrorSpec spec(/*aabs=*/0);

  // We pass -NaN in the input, but the output might contain +NaN instead.  This
  // is fine; we don't gurantee that XLA preserves the body of NaNs.
  spec.all_nans_are_equivalent = true;

  EXPECT_TRUE(
      RunAndCompare(std::move(module), {&scatter_indices, &updates}, spec));
}

INSTANTIATE_TEST_SUITE_P(EdgeCases, ScatterEdgeCaseTestP,
                         ::testing::Values("minimum", "maximum", "add"));

}  // namespace
}  // namespace xla
