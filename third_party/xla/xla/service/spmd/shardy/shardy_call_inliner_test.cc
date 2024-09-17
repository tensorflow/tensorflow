/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/shardy_call_inliner.h"

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace sdy {

using ShardyCallInlinerTest = xla::HloTestBase;

TEST_F(ShardyCallInlinerTest, MhloToHloShmapBodyNotInlined) {
  const char* const hloString = R"(
    HloModule jit_f, entry_computation_layout={(f32[8,8]{1,0})->f32[8,8]{1,0}}

    %prefix_shmap_body_suffix.4 (Arg_0.5: f32[1,8]) -> f32[1,8] {
      %Arg_0.5 = f32[1,8]{1,0} parameter(0)
      ROOT %add.6 = f32[1,8]{1,0} add(f32[1,8]{1,0} %Arg_0.5, f32[1,8]{1,0} %Arg_0.5), metadata={source_file="-" source_line=11}
    }

    ENTRY %main.10 (Arg_0.1: f32[8,8]) -> f32[8,8] {
      %Arg_0.1 = f32[8,8]{1,0} parameter(0)
      %custom-call.2 = f32[8,8]{1,0} custom-call(f32[8,8]{1,0} %Arg_0.1), custom_call_target="Sharding", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=3}
      %custom-call.3 = f32[1,8]{1,0} custom-call(f32[8,8]{1,0} %custom-call.2), custom_call_target="SPMDFullToShardShape", sharding={manual}, metadata={source_file="-" source_line=4}
      %call.7 = f32[1,8]{1,0} call(f32[1,8]{1,0} %custom-call.3), to_apply=%prefix_shmap_body_suffix.4
      %custom-call.8 = f32[1,8]{1,0} custom-call(f32[1,8]{1,0} %call.7), custom_call_target="Sharding", sharding={manual}, metadata={source_file="-" source_line=6}
      ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(f32[1,8]{1,0} %custom-call.8), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=7}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hloString));
  module->mutable_config().set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardyCallInliner().Run(module.get()));
  VLOG(1) << module->ToString();
  // The single call in the module is not inlined.
  EXPECT_FALSE(changed);

  HloInstruction* call = FindInstruction(module.get(), xla::HloOpcode::kCall);
  EXPECT_NE(call, nullptr);
  EXPECT_TRUE(call->has_to_apply());
  EXPECT_EQ(call->to_apply()->name(), "prefix_shmap_body_suffix.4");
}

}  // namespace sdy
}  // namespace xla
