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

#include "xla/service/gather_expander.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class GatherExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  void CheckWhileBody(HloModule* module, absl::string_view expected) {
    std::vector<HloInstruction*> while_instructions =
        FindInstructions(module, HloOpcode::kWhile);
    EXPECT_EQ(while_instructions.size(), 1);
    HloComputation* while_body = while_instructions[0]->while_body();
    EXPECT_TRUE(*RunFileCheck(
        while_body->ToString(
            HloPrintOptions{}.set_include_layout_in_shapes(false)),
        expected));
  }
};

TEST_F(GatherExpanderTest, ErrorStatusOnTooManyIndices) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2147483647,5] parameter(1)
  ROOT gather = s32[2147483647,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  absl::Status status = GatherExpander{GatherExpander::kEliminateAllGathers}
                            .Run(module.get())
                            .status();
  EXPECT_EQ(status.code(), tsl::error::UNIMPLEMENTED);

  ASSERT_THAT(
      status.message(),
      ::testing::HasSubstr("Gather operations with more than 2147483647 gather "
                           "indices are not supported."));
}

TEST_F(GatherExpanderTest, AvoidDegenerateDims) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = nullptr;
  for (auto* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(while_instr, nullptr)
          << "Expected exactly one while instruction in the entry computation "
             "after gather expansion";
      while_instr = instr;
    }
  }

  ASSERT_NE(while_instr, nullptr)
      << "Expected exactly one while instruction in the entry computation "
         "after gather expansion";

  // We want to avoid create while loop with shapes that have degenerate
  // dimensions for TF gather.  In this case we expect the loop state to be of
  // the shape (sNN[], s32[3,3]{1,0}, s32[2]{0}, s32[2,3]{1,0}).  The leading
  // sNN is an implementation detail from WhileUtil::MakeCountedLoop so we don't
  // check it here (though in theory the form of the while loop state is itself
  // an implementation detail from WhileUtil::MakeCountedLoop).

  const Shape& while_shape = while_instr->shape();
  ASSERT_TRUE(while_shape.IsTuple());
  ASSERT_EQ(ShapeUtil::TupleElementCount(while_shape), 4);

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {3, 3}),
      ShapeUtil::GetTupleElementShape(while_shape, 1)));

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {2}),
      ShapeUtil::GetTupleElementShape(while_shape, 2)));

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {2, 3}),
      ShapeUtil::GetTupleElementShape(while_shape, 3)));
}

TEST_F(GatherExpanderTest, CheckOpMetadata) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  OpMetadata metadata;
  metadata.set_op_name("Gather");
  module->entry_computation()->root_instruction()->set_metadata(metadata);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = nullptr;
  for (auto* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(while_instr, nullptr)
          << "Expected exactly one while instruction in the entry computation "
             "after gather expansion";
      while_instr = instr;
    }
  }

  ASSERT_NE(while_instr, nullptr)
      << "Expected exactly one while instruction in the entry computation "
         "after gather expansion";
  EXPECT_EQ(while_instr->metadata().op_name(), "Gather");
}

TEST_F(GatherExpanderTest, EliminateSimpleGathersSkipsNontrivialGather) {
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_FALSE(changed);
}

TEST_F(GatherExpanderTest, EliminateSimpleGathersRewritesTrivialGather) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[100] parameter(0)
  indices = s32[1] parameter(1)
  ROOT gather = s32[10] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={},
      start_index_map={0},
      index_vector_dim=0,
      slice_sizes={10}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateAllGathers);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
}

TEST_F(GatherExpanderTest, GatherIsBroadcast) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[1,3] parameter(0)
  indices = s32[7,5] parameter(1)
  ROOT gather = s32[7,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,3}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
  ASSERT_TRUE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                 {HloOpcode::kBroadcast}));
  module->VerifyOrAddFailure("after-gather-expander.");
}

TEST_F(GatherExpanderTest, GatherIsBroadcastBatchDim) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[1,3,1] parameter(0)
  indices = s32[1,5] parameter(1)
  ROOT gather = s32[1,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={2},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,3,1},
      operand_batching_dims={0},
      start_indices_batching_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
  ASSERT_TRUE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                 {HloOpcode::kBroadcast}));
  module->VerifyOrAddFailure("after-gather-expander.");
}

TEST_F(GatherExpanderTest, GatherToLoopWithBatchDims) {
  const std::string hlo_text = R"(
HloModule GatherWithBatchDims

ENTRY main {
  operand = s32[5,2] parameter(0)
  indices = s32[5,1] parameter(1)
  ROOT gather = s32[5,1] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={},
      start_index_map={1},
      index_vector_dim=1,
      operand_batching_dims={0},
      start_indices_batching_dims={0},
      slice_sizes={1,1}
}
)";
  const std::string expected = R"(
  //CHECK: (s32[], s32[5,2], s32[5,1], s32[5,1])) -> (s32[], s32[5,2], s32[5,1], s32[5,1]) {
  //CHECK: %[[PARAM:.*]] = (s32[], s32[5,2], s32[5,1], s32[5,1]) parameter(0)
  //CHECK: %[[I:.*]] = s32[] get-tuple-element(%[[PARAM]]), index=
  //CHECK: %[[CONSTANT1:.*]] = s32[] constant(1)
  //CHECK: %[[I_PLUS_1:.*]] = s32[] add(%[[I]], %[[CONSTANT1]])
  //CHECK: %[[OPERAND:.*]] = s32[5,2] get-tuple-element(%[[PARAM]]), index=1
  //CHECK: %[[START_INDICES:.*]] = s32[5,1] get-tuple-element(%[[PARAM]]), index=2
  //CHECK: %[[RESULT:.*]] = s32[5,1] get-tuple-element(%[[PARAM]]), index=3

  //CHECK: %[[I_1D_1:.*]] = s32[1] broadcast(%[[I]])
  //CHECK: %[[I_1D_2:.*]] = s32[1] broadcast(%[[I]])

  //CHECK: %[[START_INDICES_INDEX_D1_PAD:.*]] = s32[] constant(0)
  //CHECK: %[[START_INDICES_INDEX_VECTOR:.*]] = s32[2] pad(%[[I_1D_2]], %[[START_INDICES_INDEX_D1_PAD]]), padding=0_1
  //CHECK: %[[START_INDICES_INDEX_D0_SLICE:.*]] = s32[1] slice(%[[START_INDICES_INDEX_VECTOR]]), slice={[0:1]}
  //CHECK: %[[START_INDICES_INDEX_D0:.*]] = s32[] reshape(%[[START_INDICES_INDEX_D0_SLICE]])
  //CHECK: %[[START_INDICES_INDEX_D1_SLICE:.*]] = s32[1] slice(%[[START_INDICES_INDEX_VECTOR]]), slice={[1:2]}
  //CHECK: %[[START_INDICES_INDEX_D1:.*]] = s32[] reshape(%[[START_INDICES_INDEX_D1_SLICE]])
  //CHECK: %[[INDEX_VECTOR:.*]] = s32[1,1] dynamic-slice(%[[START_INDICES]], %[[START_INDICES_INDEX_D0]], %[[START_INDICES_INDEX_D1]])

  //CHECK: %[[OFFSET_RAW:.*]] = s32[1] reshape(%[[INDEX_VECTOR]])
  //CHECK: %[[OFFSET:.*]] = s32[1] slice(%[[OFFSET_RAW]])
  //CHECK: %[[OPERAND_INDEX:.*]] = s32[2] concatenate(%[[I_1D_1]], %[[OFFSET]])
  //CHECK: %[[OPERAND_INDEX_D0_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[0:1]}
  //CHECK: %[[OPERAND_INDEX_D0:.*]] = s32[] reshape(%[[OPERAND_INDEX_D0_RAW]])
  //CHECK: %[[OPERAND_INDEX_D1_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[1:2]}
  //CHECK: %[[OPERAND_INDEX_D1:.*]] = s32[] reshape(%[[OPERAND_INDEX_D1_RAW]])
  //CHECK: %[[RESULT_SLICE_RAW0:.*]] = s32[1,1] dynamic-slice(%[[OPERAND]], %[[OPERAND_INDEX_D0]], %[[OPERAND_INDEX_D1]])

  //CHECK: %[[RESULT_SLICE_RAW1:.*]] = s32[1] reshape(%[[RESULT_SLICE_RAW0]])
  //CHECK: %[[RESULT_SLICE:.*]] = s32[1,1] reshape(%[[RESULT_SLICE_RAW1]])
  //CHECK: %[[RESULT_INDEX_D1_PAD:.*]] = s32[] constant(0)
  //CHECK: %[[RESULT_INDEX_VECTOR:.*]] = s32[2] pad(%[[I_1D_2]], %[[RESULT_INDEX_D1_PAD]]), padding=0_1
  //CHECK: %[[RESULT_INDEX_D0_SLICE:.*]] = s32[1] slice(%[[RESULT_INDEX_VECTOR]]), slice={[0:1]}
  //CHECK: %[[RESULT_INDEX_D0:.*]] = s32[] reshape(%[[RESULT_INDEX_D0_SLICE]])
  //CHECK: %[[RESULT_INDEX_D1_SLICE:.*]] = s32[1] slice(%[[RESULT_INDEX_VECTOR]]), slice={[1:2]}
  //CHECK: %[[RESULT_INDEX_D1:.*]] = s32[] reshape(%[[RESULT_INDEX_D1_SLICE]])
  //CHECK: %[[UPDATED_RESULT:.*]] = s32[5,1] dynamic-update-slice(%[[RESULT]], %[[RESULT_SLICE]], %[[RESULT_INDEX_D0]], %[[RESULT_INDEX_D1]])

  //CHECK: ROOT %{{.*}} = (s32[], s32[5,2], s32[5,1], s32[5,1]) tuple(%[[I_PLUS_1]], %[[OPERAND]], %[[START_INDICES]], %[[UPDATED_RESULT]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);
  CheckWhileBody(module.get(), expected);
}

TEST_F(GatherExpanderTest, GatherToLoopWithBatchDimsAndCollapsedDims) {
  const std::string hlo_text = R"(
HloModule GatherWithBatchAndCollapsedDims

ENTRY main {
  operand = s32[7,3,4,5] parameter(0)
  indices = s32[5,2,7] parameter(1)
  ROOT gather = s32[5,3,2,7] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={2},
      start_index_map={2},
      index_vector_dim=3,
      operand_batching_dims={0,3},
      start_indices_batching_dims={2,0},
      slice_sizes={1,3,1,1}
}
)";
  // Compared with the previous test, this test adds complexity in calculating
  // the indices for the operand. As such, we mostly check the operand indices
  // here.
  const std::string expected = R"(
  //CHECK: (s32[], s32[7,3,4,5], s32[70], s32[70,3])) -> (s32[], s32[7,3,4,5], s32[70], s32[70,3]) {
  //CHECK: %[[PARAM:.*]] = (s32[], s32[7,3,4,5], s32[70], s32[70,3]) parameter(0)
  //CHECK: %[[I:.*]] = s32[] get-tuple-element(%[[PARAM]]), index=0

  //CHECK: %[[CONSTANT1:.*]] = s32[] constant(1)
  //CHECK: %[[I_PLUS_1:.*]] = s32[] add(%[[I]], %[[CONSTANT1]])
  //CHECK: %[[OPERAND:.*]] = s32[7,3,4,5] get-tuple-element(%[[PARAM]]), index=1
  //CHECK: %[[START_INDICES:.*]] = s32[70] get-tuple-element(%[[PARAM]]), index=2

  //CHECK: %[[CONSTANT7:.*]] = s32[] constant(7)
  //CHECK: %[[BD0_RAW:.*]] = s32[] remainder(%[[I]], %[[CONSTANT7]])
  //CHECK: %[[BD0:.*]] = s32[1] broadcast(%[[BD0_RAW]])
  //CHECK: %[[CONSTANT0:.*]] = s32[1] constant({0})
  //CHECK: %[[I_1D_1:.*]] = s32[1] broadcast(%[[I]])
  //CHECK: %[[START_INDICES_INDEX_RAW:.*]] = s32[1] slice(%[[I_1D_1]])
  //CHECK: %[[START_INDICES_INDEX:.*]] = s32[] reshape(%[[START_INDICES_INDEX_RAW]])
  //CHECK: %[[INDEX_VECTOR:.*]] = s32[1] dynamic-slice(%[[START_INDICES]], %[[START_INDICES_INDEX]])

  //CHECK: %[[OFFSET:.*]] = s32[1] slice(%[[INDEX_VECTOR]])
  //CHECK: %[[BD1:.*]] = s32[] divide(%[[I]], %[[CONSTANT7]])
  //CHECK: %[[CONSTANT2:.*]] = s32[] constant(2)
  //CHECK: %[[BD2_RAW:.*]] = s32[] divide(%[[BD1]], %[[CONSTANT2]])
  //CHECK: %[[BD2:.*]] = s32[1] broadcast(%[[BD2_RAW]])
  //CHECK: %[[OPERAND_INDEX:.*]] = s32[4] concatenate(%[[BD0]], %[[CONSTANT0]], %[[OFFSET]], %[[BD2]])

  //CHECK: %[[OPERAND_INDEX_D0_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[0:1]}
  //CHECK: %[[OPERAND_INDEX_D0:.*]] = s32[] reshape(%[[OPERAND_INDEX_D0_RAW]])
  //CHECK: %[[OPERAND_INDEX_D1_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[1:2]}
  //CHECK: %[[OPERAND_INDEX_D1:.*]] = s32[] reshape(%[[OPERAND_INDEX_D1_RAW]])
  //CHECK: %[[OPERAND_INDEX_D2_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[2:3]}
  //CHECK: %[[OPERAND_INDEX_D2:.*]] = s32[] reshape(%[[OPERAND_INDEX_D2_RAW]])
  //CHECK: %[[OPERAND_INDEX_D3_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[3:4]}
  //CHECK: %[[OPERAND_INDEX_D3:.*]] = s32[] reshape(%[[OPERAND_INDEX_D3_RAW]])
  //CHECK: %{{.*}} = s32[1,3,1,1] dynamic-slice(%[[OPERAND]], %[[OPERAND_INDEX_D0]], %[[OPERAND_INDEX_D1]], %[[OPERAND_INDEX_D2]], %[[OPERAND_INDEX_D3]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);
  CheckWhileBody(module.get(), expected);
}

}  // namespace
}  // namespace xla
