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

#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {
TEST(GatherExpanderTest, ErrorStatusOnTooManyIndices) {
  const string hlo_text = R"(
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
                          ParseHloString(hlo_text));

  Status status = GatherExpander{}.Run(module.get()).status();
  EXPECT_EQ(status.code(), tensorflow::error::UNIMPLEMENTED);

  ASSERT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Gather operations with more than 2147483647 gather "
                           "indices are not supported."));
}

TEST(GatherExpanderTest, AvoidDegenerateDims) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, GatherExpander{}.Run(module.get()));
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
  ASSERT_TRUE(ShapeUtil::IsTuple(while_shape));
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
}  // namespace
}  // namespace xla
