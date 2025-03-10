/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/scatter_simplifier.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class ScatterSimplifierTest : public HloTestBase {};

TEST_F(ScatterSimplifierTest, InsertsIndexVectorAndWindowDims) {
  // Verifies that ScatterSimplifier
  // - Makes the index_vector_dim dimensions explicit
  // - Inserts inserted_window_dims into updates.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT tuple = tuple(p2, p3)
    }

    ENTRY kernel_entry {
      operand0 = f32[3,3] parameter(0)
      operand1 = f32[3,3] parameter(1)
      indices = s32[2] parameter(2)
      update0 = f32[2,3] parameter(3)
      update1 = f32[2,3] parameter(4)
      ROOT scatter = (f32[3,3], f32[3,3]) scatter(operand0, operand1, indices,
                                                  update0, update1),
          to_apply=scatter_computation,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
      CHECK: %[[SCATTER_DIMS_WITH_VECTOR:.*]] = s32[2,1]{1,0} reshape(%indices)
      CHECK: %[[RESHAPED_UPDATES0:.*]] = f32[2,1,3]{2,1,0} reshape(%update0)
      CHECK: %[[RESHAPED_UPDATES1:.*]] = f32[2,1,3]{2,1,0} reshape(%update1)
      CHECK: ROOT %scatter = (f32[3,3]{1,0}, f32[3,3]{1,0}) scatter(
      CHECK-SAME:   %operand0, %operand1, %[[SCATTER_DIMS_WITH_VECTOR]],
      CHECK-SAME:   %[[RESHAPED_UPDATES0]], %[[RESHAPED_UPDATES1]]),
      CHECK-SAME: update_window_dims={1,2},
      CHECK-SAME: inserted_window_dims={},
      CHECK-SAME: scatter_dims_to_operand_dims={0},
      CHECK-SAME: index_vector_dim=1,
      CHECK-SAME: to_apply=%scatter_computation
  )");
}

TEST_F(ScatterSimplifierTest, CollapsesScatterDims) {
  // Verifies that ScatterSimplifier collapses multiple scatter dimensions into
  // one.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3,3] parameter(0)
      indices = s32[2,1,2] parameter(1)
      update = f32[2,1,1,3] parameter(2)
      ROOT scatter = f32[3,3] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={2, 3},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0,1},
          index_vector_dim=2
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
           CHECK: %[[RESHAPED_INDICES:.*]] = s32[2,2]{1,0} reshape(%indices)
           CHECK: %[[RESHAPED_UPDATES:.*]] = f32[2,1,3]{2,1,0} reshape(%update)
           CHECK: scatter(
      CHECK-SAME: %[[RESHAPED_INDICES]]
      CHECK-SAME: %[[RESHAPED_UPDATES]]
  )");
}

TEST_F(ScatterSimplifierTest, NoOpForSimpleScatter) {
  // Verifies that ScatterSimplifier does nothing if the scatter is already
  // simple.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3,3] parameter(0)
      indices = s32[2,2] parameter(1)
      update = f32[2,1,3] parameter(2)
      ROOT scatter = f32[3,3] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0,1},
          index_vector_dim=1
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), std::nullopt);
}

TEST_F(ScatterSimplifierTest, MovesIndexVectorDim) {
  // Verifies that ScatterSimplifier makes index_vector_dim trailing.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3,3] parameter(0)
      indices = s32[2,1] parameter(1)
      update = f32[1,3,3] parameter(2)
      ROOT scatter = f32[3,3] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1, 2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0,1},
          index_vector_dim=0
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
           CHECK: %[[TRANSPOSED_INDICES:.*]] = s32[1,2]{1,0}
      CHECK-SAME:     transpose(%indices), dimensions={1,0}
           CHECK: scatter(%operand, %[[TRANSPOSED_INDICES]], %update),
      CHECK-SAME:     index_vector_dim=1
  )");
}

TEST_F(ScatterSimplifierTest, TransformsUpdatesAndOperandUsingScatterDims) {
  // Verifies that ScatterSimplifier transposes updates and operands to conform
  // to scatter_dims_to_operand_dims.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3,4,5] parameter(0)
      indices = s32[2,2] parameter(1)
      update = f32[2,1,1,3] parameter(2)
      ROOT scatter = f32[3,4,5] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1, 2, 3},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={2,0},
          index_vector_dim=1
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
           CHECK: %[[T_OPERAND:.*]] = f32[5,3,4]{2,1,0} transpose(%operand),
      CHECK-SAME:     dimensions={2,0,1}
           CHECK: %[[T_UPDATES:.*]] = f32[2,3,1,1]{3,2,1,0} transpose(%update),
      CHECK-SAME:     dimensions={0,3,1,2}
           CHECK: %[[SCATTER:.*]] = {{.*}} scatter(
      CHECK-SAME:     %[[T_OPERAND]], %indices, %[[T_UPDATES]])
      CHECK-SAME:     scatter_dims_to_operand_dims={0,1},
           CHECK: ROOT %{{.*}} = f32[3,4,5]
      CHECK-SAME:     transpose(%[[SCATTER]]), dimensions={1,2,0}
  )");
}

TEST_F(ScatterSimplifierTest, MakesScatterDimensionsLeadingInUpdates) {
  // Verifies that ScatterSimplifier moves the scatter dimensions in updates.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3] parameter(0)
      indices = s32[1,1] parameter(1)
      update = f32[2,1] parameter(2)
      ROOT scatter = f32[3] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={0},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
           CHECK: %[[TRANSPOSED_UPDATES:.*]] = f32[1,2]{1,0}
      CHECK-SAME:     transpose(%update), dimensions={1,0}
           CHECK: scatter(
      CHECK-SAME:     %[[TRANSPOSED_UPDATES]]
      CHECK-SAME:     update_window_dims={1},
  )");
}

TEST_F(ScatterSimplifierTest, ZeroDimScatterIndices) {
  // Verifies that zero-dimensional scatter indices are transformed correctly.
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }
    ENTRY kernel_entry {
      operand = f32[4,4] parameter(0)
      indices = s32[2] parameter(1)
      update = f32[3,3] parameter(2)
      ROOT scatter = f32[4,4]{1,0} scatter(operand, indices, update),
          update_window_dims={0,1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0,1},
          index_vector_dim=0,
          to_apply=scatter_computation
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
      CHECK: scatter(
    )");
}

TEST_F(ScatterSimplifierTest,
       IsSimplifiedScatterReturnsFalseForUnsortedWindowDims) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      %p0 = f32[] parameter(0)
      ROOT result = f32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = f32[3,2] parameter(0)
      indices = s32[1,1] parameter(1)
      update = f32[1,2,2] parameter(2)
      ROOT scatter = f32[3,2] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={2,1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";
  auto module = ParseAndReturnUnverifiedModule(kModuleStr).value();
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_FALSE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
}

TEST_F(ScatterSimplifierTest, ScatterIntoScalar) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[] parameter(0)
      indices = s32[0]{0} parameter(1)
      updates = s32[] parameter(2)
      ROOT scatter = s32[] scatter(operand, indices, updates),
          update_window_dims={},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={},
          index_vector_dim=0,
          to_apply=scatter_computation
    }
  )";
  auto module = ParseAndReturnUnverifiedModule(kModuleStr).value();
  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
    CHECK: ENTRY
    CHECK: %[[OPERAND:.*]] = s32[] parameter(0)
    CHECK: %[[UPDATES:.*]] = s32[] parameter(2)
    CHECK: ROOT %{{.*}} = s32[] add(%[[OPERAND]], %[[UPDATES]])
  )");
}

TEST_F(ScatterSimplifierTest, VariadicScatterIntoScalar) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule scatter_simplifier

    scatter_computation {
      p0 = f32[] parameter(0)
      p1 = bf16[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = bf16[] parameter(3)
      ROOT tuple = tuple(p2, p3)
    }

    ENTRY kernel_entry {
      operand0 = f32[] parameter(0)
      operand1 = bf16[] parameter(1)
      indices = s32[0]{0} parameter(2)
      updates0 = f32[] parameter(3)
      updates1 = bf16[] parameter(4)
      ROOT scatter = (f32[], bf16[]) scatter(operand0, operand1, indices, updates0, updates1),
          update_window_dims={},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={},
          index_vector_dim=0,
          to_apply=scatter_computation
    })";

  RunAndFilecheckHloRewrite(kModuleStr, ScatterSimplifier(), R"(
    CHECK: ENTRY
    CHECK: %[[UPDATES0:.*]] = f32[] parameter(3)
    CHECK: %[[UPDATES1:.*]] = bf16[] parameter(4)
    CHECK: ROOT %{{.*}} = (f32[], bf16[]) tuple(%[[UPDATES0]], %[[UPDATES1]])
  )");
}

class SimpleScatterExampleTest : public HloTestBase {};

TEST_F(SimpleScatterExampleTest, 1x1d) {
  constexpr absl::string_view hlo_text = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[4] parameter(0)
      indices = s32[1,1] parameter(1)
      update = s32[1,2] parameter(2)
      ROOT scatter = s32[4] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{2}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{2, 3}});
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module, {&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<int32_t>({1, 2, 5, 7}), result));
}

TEST_F(SimpleScatterExampleTest, 2x2d) {
  constexpr absl::string_view hlo_text = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[2,2] parameter(0)
      indices = s32[2,2] parameter(1)
      update = s32[2,2,2] parameter(2)
      ROOT scatter = s32[2,2] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0,1},
          index_vector_dim=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module, {&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>(
          {{1 + 10 + 50, 2 + 20 + 60}, {3 + 30 + 70, 4 + 40 + 80}}),
      result));
}

TEST_F(SimpleScatterExampleTest, 2x2d_n_less_than_N) {
  constexpr absl::string_view hlo_text = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[2,2] parameter(0)
      indices = s32[2,1] parameter(1)
      update = s32[2,2,2] parameter(2)
      ROOT scatter = s32[2,2] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0}, {0}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module, {&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>(
          {{1 + 10 + 50, 2 + 20 + 60}, {3 + 30 + 70, 4 + 40 + 80}}),
      result));
}

TEST_F(SimpleScatterExampleTest, 2x2d_nofit) {
  constexpr absl::string_view hlo_text = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[2,2] parameter(0)
      indices = s32[2,1] parameter(1)
      update = s32[2,2,2] parameter(2)
      ROOT scatter = s32[2,2] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});

  // Note: the 2nd update slice won't fit.
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0}, {2}});

  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module, {&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1 + 10, 2 + 20}, {3 + 30, 4 + 40}}),
      result));
}

TEST_F(SimpleScatterExampleTest, 3x1d) {
  constexpr absl::string_view hlo_text = R"(
    HloModule scatter_simplifier

    scatter_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY kernel_entry {
      operand = s32[4] parameter(0)
      indices = s32[3,1] parameter(1)
      update = s32[3,2] parameter(2)
      ROOT scatter = s32[4] scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto scatter = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ScatterSimplifier::IsSimplifiedScatter(
      Cast<HloScatterInstruction>(scatter)));
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0}, {1}, {2}});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {10, 20}, {200, 300}});
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module, {&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(
                                 {1 + 1, 2 + 2 + 10, 3 + 20 + 200, 4 + 300}),
                             result));
}

TEST_F(SimpleScatterExampleTest,
       ScatterSimplifierPropagatesPackingForSubByteType) {
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule scatter_simplifier

    scatter_computation {
      p0 = s4[] parameter(0)
      ROOT result = s4[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = s4[3,4,5]{2,1,0$0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      update = s4[2,1,1,3]{3,2,1,0$0} parameter(2)
      ROOT scatter = s4[3,4,5]{2,1,0$0} scatter(operand, indices, update),
          to_apply=scatter_computation,
          update_window_dims={1, 2, 3},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={2,0},
          index_vector_dim=1
    })";

  // First, test with no packing.
  std::string unpacked_module_str = absl::Substitute(kHloTextTemplate, "");
  RunAndFilecheckHloRewrite(unpacked_module_str, ScatterSimplifier(), R"(
           CHECK: %[[T_OPERAND:.*]] = s4[5,3,4]{2,1,0} transpose(%operand),
      CHECK-SAME:     dimensions={2,0,1}
           CHECK: %[[T_UPDATES:.*]] = s4[2,3,1,1]{3,2,1,0} transpose(%update),
      CHECK-SAME:     dimensions={0,3,1,2}
           CHECK: %[[SCATTER:.*]] = {{.*}} scatter(
      CHECK-SAME:     %[[T_OPERAND]], %indices, %[[T_UPDATES]])
      CHECK-SAME:     scatter_dims_to_operand_dims={0,1},
           CHECK: ROOT %{{.*}} = s4[3,4,5]{2,1,0}
      CHECK-SAME:     transpose(%[[SCATTER]]), dimensions={1,2,0}
  )");

  // Second, test with packing.
  std::string packed_module_str = absl::Substitute(kHloTextTemplate, ":E(4)");
  RunAndFilecheckHloRewrite(packed_module_str, ScatterSimplifier(), R"(
           CHECK: %[[T_OPERAND:.*]] = s4[5,3,4]{2,1,0:E(4)} transpose(%operand),
      CHECK-SAME:     dimensions={2,0,1}
           CHECK: %[[T_UPDATES:.*]] = s4[2,3,1,1]{3,2,1,0:E(4)} transpose(%update),
      CHECK-SAME:     dimensions={0,3,1,2}
           CHECK: %[[SCATTER:.*]] = {{.*}} scatter(
      CHECK-SAME:     %[[T_OPERAND]], %indices, %[[T_UPDATES]])
      CHECK-SAME:     scatter_dims_to_operand_dims={0,1},
           CHECK: ROOT %{{.*}} = s4[3,4,5]{2,1,0:E(4)}
      CHECK-SAME:     transpose(%[[SCATTER]]), dimensions={1,2,0}
  )");
}

}  // namespace
}  // namespace xla
