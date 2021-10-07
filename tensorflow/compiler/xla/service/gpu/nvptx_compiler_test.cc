/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

using NVPTXCompilerTest = HloTestBase;

TEST_F(NVPTXCompilerTest, AllReducePerformedInplace) {
  const absl::string_view hlo_string = R"(
HloModule Module, input_output_alias={ {}: (0, {}, may-alias) }

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0)
  ROOT allreduce = f32[128] all-reduce(param0),
    replica_groups={}, to_apply=summit
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  NVPTXCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          compiler.AssignBuffers(module.get()));

  HloInstruction* all_reduce = module->entry_computation()->root_instruction();
  EXPECT_TRUE(buffer_assignment->SharesTopLevelSlice(all_reduce,
                                                     all_reduce->operand(0)));
}

TEST_F(NVPTXCompilerTest, AllReducePerformedInplaceTwoOperands) {
  const absl::string_view hlo_string = R"(
HloModule Module,
  input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0)
  param1 = f32[128] parameter(1)
  ROOT allreduce = (f32[128], f32[128]) all-reduce(param0, param1),
    replica_groups={}, to_apply=summit
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  NVPTXCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          compiler.AssignBuffers(module.get()));

  HloInstruction* all_reduce = module->entry_computation()->root_instruction();
  EXPECT_TRUE(buffer_assignment->SharesSliceAtIndex(
      all_reduce, {0}, all_reduce->operand(0), {}));
  EXPECT_TRUE(buffer_assignment->SharesSliceAtIndex(
      all_reduce, {1}, all_reduce->operand(1), {}));
}

}  // namespace gpu
}  // namespace xla
