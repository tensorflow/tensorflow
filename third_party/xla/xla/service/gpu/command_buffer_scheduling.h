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

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
      HloInstructionSequence inst_sequence,
      std::function<bool(const HloInstruction*)> is_command);
  static void MoveParametersToFront(HloComputation* computation);

  struct BuildCommandBufferResult {
    std::unique_ptr<HloComputation> computation;

    // Maps external instructions used by the command buffer to a parameter
    // of the command buffer computation. The command buffer uses parameters
    // to access the results of external instructions.
    absl::flat_hash_map<HloInstruction*, HloParameterInstruction*>
        parameters_map;

    // We move some instructions to the command buffer computation and return
    // the results back to the original computation by tuple. This field maps
    // the original instruction to the tuple index of the result that replaces
    // the original instruction.
    absl::flat_hash_map<HloInstruction*, int64_t> inst_to_tuple_index_map;

    // Map original instructions to their clones in the command buffer
    // computation.
    absl::flat_hash_map<HloInstruction*, HloInstruction*> instructions_map;
  };

  // Builds a computation from the instruction sequence. Used values constructed
  // by instructions outside of the sequence are passed in as parameters.
  // Results of instructions in the sequence are returned in a tuple.
  static StatusOr<BuildCommandBufferResult> BuildCommandBuffer(
      HloInstructionSequence seq);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_COMMAND_BUFFER_SCHEDULING_H_
