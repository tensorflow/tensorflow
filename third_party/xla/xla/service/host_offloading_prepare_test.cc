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

#include "xla/service/host_offloading_prepare.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class HostOffloadingPrepareTest : public HloTestBase {
 protected:
  absl::StatusOr<bool> RunPass(HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (module->has_schedule()) {
      return absl::InternalError("Expected a non-scheduled module");
    }
    HostOffloadingPrepare pass;
    TF_ASSIGN_OR_RETURN(bool changed, pass.Run(module));
    return changed;
  }

  std::vector<const HloInstruction*> GetHostOffloadAsyncStartInstructions(
      const HloModule* module) {
    std::vector<const HloInstruction*> result;
    for (const HloComputation* computation : module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kAsyncStart &&
            instruction->async_execution_thread() ==
                HloInstruction::kHostThread) {
          result.push_back(instruction);
        }
      }
    }
    return result;
  }
};

TEST_F(HostOffloadingPrepareTest, SingleInputHasMoveToHost) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.0)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.1), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_TRUE(changed);

  for (const HloInstruction* instruction :
       GetHostOffloadAsyncStartInstructions(module.get())) {
    // None of the inputs should be a "to host" custom call.
    for (const HloInstruction* operand : instruction->operands()) {
      EXPECT_FALSE(operand->IsCustomCall(
          {host_memory_offload_annotations::kMoveToHostCustomCallTarget}));
    }
    // None of the outputs should be a "to device" custom call.
    for (const HloInstruction* user : instruction->users()) {
      EXPECT_FALSE(user->IsCustomCall(
          {host_memory_offload_annotations::kMoveToDeviceCustomCallTarget}));
    }
  }
}

TEST_F(HostOffloadingPrepareTest, MultipleInputHasOneMoveToHost) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  Arg_0.1 = s32[32]{0} parameter(1)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.1)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  param_1 = s32[32]{0} parameter(1)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0, param_1), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}, s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.1, custom-call.1), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_TRUE(changed);

  for (const HloInstruction* instruction :
       GetHostOffloadAsyncStartInstructions(module.get())) {
    // None of the inputs should be a "to host" custom call.
    for (const HloInstruction* operand : instruction->operands()) {
      EXPECT_FALSE(operand->IsCustomCall(
          {host_memory_offload_annotations::kMoveToHostCustomCallTarget}));
    }
    // None of the outputs should be a "to device" custom call.
    for (const HloInstruction* user : instruction->users()) {
      EXPECT_FALSE(user->IsCustomCall(
          {host_memory_offload_annotations::kMoveToDeviceCustomCallTarget}));
    }
  }
}

TEST_F(HostOffloadingPrepareTest, MultipleInputHasMultipleMoveToHost) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  Arg_0.1 = s32[32]{0} parameter(1)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.1)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  param_1 = s32[32]{0} parameter(1)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0, param_1), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.2 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}, s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.1, custom-call.2), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_TRUE(changed);

  for (const HloInstruction* instruction :
       GetHostOffloadAsyncStartInstructions(module.get())) {
    // None of the inputs should be a "to host" custom call.
    for (const HloInstruction* operand : instruction->operands()) {
      EXPECT_FALSE(operand->IsCustomCall(
          {host_memory_offload_annotations::kMoveToHostCustomCallTarget}));
    }
    // None of the outputs should be a "to device" custom call.
    for (const HloInstruction* user : instruction->users()) {
      EXPECT_FALSE(user->IsCustomCall(
          {host_memory_offload_annotations::kMoveToDeviceCustomCallTarget}));
    }
  }
}

TEST_F(HostOffloadingPrepareTest, SingleInputHasMoveToDevice) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.0)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.2 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToDevice"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.2), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(HostOffloadingPrepareTest, MultipleInputHasOneMoveToDevice) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  Arg_0.1 = s32[32]{0} parameter(1)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.1)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  param_1 = s32[32]{0} parameter(1)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0, param_1), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.2 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToDevice"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}, s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.2, custom-call.2), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(HostOffloadingPrepareTest, MultipleInputHasMultipleMoveToDevice) {
  const std::string& hlo_string = R"(
HloModule my_module, entry_computation_layout={(s32[32]{0:T(128)})->s32[32]{0:T(128)}}

host_computation {
  Arg_0.0 = s32[32]{0} parameter(0)
  Arg_0.1 = s32[32]{0} parameter(1)
  ROOT multiply.0 = s32[32]{0} multiply(Arg_0.0, Arg_0.1)
}, execution_thread="host"

async_computation {
  param_0 = s32[32]{0} parameter(0)
  param_1 = s32[32]{0} parameter(1)
  ROOT custom-call.cloned = s32[32]{0} custom-call(param_0, param_1), custom_call_target="HostExecute", called_computations={host_computation}, frontend_attributes={_xla_compute_type="host"}
}, execution_thread="host"

ENTRY main {
  Arg_0.1 = s32[32]{0:T(128)} parameter(0)
  constant.2 = s32[]{:T(128)} constant(2)
  broadcast.3 = s32[32]{0:T(128)} broadcast(constant.2), dimensions={}
  multiply.4 = s32[32]{0:T(128)} multiply(Arg_0.1, broadcast.3)
  custom-call.1 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.2 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToHost"
  custom-call.3 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToDevice"
  custom-call.4 = s32[32]{0:T(128)} custom-call(multiply.4), custom_call_target="MoveToDevice"
  custom-call.cloned.call-start = ((s32[32]{0:T(128)}, s32[32]{0:T(128)}), s32[32]{0:T(128)}, u32[]{:T(128)}) async-start(custom-call.3, custom-call.4), async_execution_thread="host", calls=async_computation
  ROOT custom-call.cloned.call-done = s32[32]{0:T(128)} async-done(custom-call.cloned.call-start), frontend_attributes={_xla_compute_type="host"}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));

  EXPECT_FALSE(changed);
}

}  // namespace

}  // namespace xla
