/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/stream_attribute_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {
static absl::StatusOr<bool> AsynchronizeInstruction(HloInstruction* instr) {
  auto instr_gpu_config = instr->backend_config<GpuBackendConfig>();
  if (!instr_gpu_config.ok() || instr_gpu_config->operation_queue_id() ==
                                    Thunk::kDefaultExecutionStreamId.value()) {
    return false;
  }
  HloComputation* computation = instr->parent();
  TF_ASSIGN_OR_RETURN(
      HloInstruction * done,
      computation->CreateAsyncInstructions(
          instr, {}, StreamAttributeAsyncWrapper::kParallelExecutionThread,
          /*replace=*/true));
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      done->backend_config<GpuBackendConfig>());
  // Set the false delay of done op to be false so it can be scheduled
  // far apart from start.
  gpu_config.set_force_earliest_schedule(false);
  TF_RETURN_IF_ERROR(done->set_backend_config(gpu_config));
  VLOG(5) << "Created async instruction: " << done->ToString();
  return true;
}
}  // namespace

absl::StatusOr<bool> StreamAttributeAsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "StreamAttributeAsyncWrapper::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (const HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool result, AsynchronizeInstruction(instr));
      changed |= result;
    }
  }
  XLA_VLOG_LINES(
      2, "StreamAttributeAsyncWrapper::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
