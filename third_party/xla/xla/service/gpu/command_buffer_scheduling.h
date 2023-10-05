/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_COMMAND_BUFFER_SCHEDULING_H_
#define XLA_SERVICE_GPU_COMMAND_BUFFER_SCHEDULING_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla::gpu {

// Lift fusion instructions to command buffers.
//
// Before the pass:
//   %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
//     ...
//   }
//
//   ENTRY %main (a: s32[], b: s32[]) -> s32[] {
//     %a = s32[] parameter(0)
//     %b = s32[] parameter(1)
//     ROOT %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop,
//       calls=%fused_computation
//   }
//
// After the pass:
//   %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
//     ...
//   }
//
//   %command_buffer (param_0: s32[], param_1: s32[]) -> s32[] {
//     %param_0 = s32[] parameter(0)
//     %param_1 = s32[] parameter(1)
//     ROOT %fusion = s32[] fusion(s32[] %param_0, s32[] %param_1), kind=kLoop,
//       calls=%fused_computation
//   }
//
//   ENTRY %main (a: s32[], b: s32[]) -> s32[] {
//     %a = s32[] parameter(0)
//     %b = s32[] parameter(1)
//     ROOT %call = s32[] call(s32[] %a, s32[] %b), to_apply=%command_buffer
//  }
//
// We currently do not have a command_buffer HLO operation, so we'll start with
// a kCall op code with an attached HLO computation. We'll consider graduating
// custom call to a first class operation later.
class CommandBufferScheduling : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "command-buffer-scheduling";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  static std::vector<HloInstructionSequence> CollectCommandBufferSequences(
      HloInstructionSequence inst_sequence);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_COMMAND_BUFFER_SCHEDULING_H_
