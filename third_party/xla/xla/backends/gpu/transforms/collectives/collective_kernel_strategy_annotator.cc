/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/collective_kernel_strategy_annotator.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

using stream_executor::gpu::AllReduceStrategy;

// Maps the runtime AllReduceStrategy enum to the proto enum.
// Both the Triton codegen kernel and the built-in custom C++ kernel use the
// same NVLink-based cost formula, so a single annotation covers both backends.
CollectiveBackendConfig::CollectiveKernelStrategy ToProtoStrategy(
    AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      return CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT;
    case AllReduceStrategy::kTwoShot:
      return CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_TWO_SHOT;
    case AllReduceStrategy::kMultimem:
      // Not yet modelled in the cost model; fall back to default.
      [[fallthrough]];
    default:
      return CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT;
  }
}

// Tries to determine the Triton kernel strategy for `instr` (which must be an
// AllReduce or AllReduceStart) and writes the result into backend_config.
// Returns true if the annotation was written.
absl::StatusOr<bool> TryAnnotateAllReduce(HloInstruction* instr,
                                          const GpuTopology& gpu_topology,
                                          bool is_multimem_enabled) {
  // Both kAllReduce and kAllReduceStart are HloAllReduceInstruction.
  const auto* all_reduce = DynCast<HloAllReduceInstruction>(instr);
  if (all_reduce == nullptr) {
    return false;
  }

  const DeviceAssignment* device_assignment = nullptr;
  if (instr->GetModule()->config().has_static_device_assignment()) {
    device_assignment =
        &instr->GetModule()->config().static_device_assignment();
  }

  absl::StatusOr<AllReduceInfo> maybe_info = BuildAllReduceInfo(
      /*is_collective_kernel_enabled=*/true, is_multimem_enabled, gpu_topology,
      all_reduce, device_assignment);
  if (absl::IsUnimplemented(maybe_info.status())) {
    VLOG(3) << "[CollectiveKernelStrategyAnnotator] Collective kernel not "
               "supported for "
            << instr->name() << ": " << maybe_info.status();
    return false;
  }
  ASSIGN_OR_RETURN(AllReduceInfo info, std::move(maybe_info));

  CollectiveBackendConfig::CollectiveKernelStrategy proto_strategy =
      ToProtoStrategy(info.all_reduce_strategy);

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr->backend_config<GpuBackendConfig>());
  gpu_config.mutable_collective_backend_config()->set_kernel_strategy(
      proto_strategy);
  RETURN_IF_ERROR(instr->set_backend_config(gpu_config));

  VLOG(3) << "[CollectiveKernelStrategyAnnotator] Annotated " << instr->name()
          << " with kernel strategy "
          << CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                 proto_strategy);
  return true;
}

// Tries to determine if the AllGather instruction should use the Triton
// collective kernel and annotates it accordingly.
// AllGather is always one-shot, so when eligible it is annotated with
// KERNEL_STRATEGY_TRITON_ONE_SHOT.  Returns true if the annotation was written.
absl::StatusOr<bool> TryAnnotateAllGather(HloInstruction* instr,
                                          const GpuTopology& gpu_topology) {
  const auto* all_gather = DynCast<HloAllGatherInstruction>(instr);
  if (all_gather == nullptr) {
    return false;
  }

  const DeviceAssignment* device_assignment = nullptr;
  if (instr->GetModule()->config().has_static_device_assignment()) {
    device_assignment =
        &instr->GetModule()->config().static_device_assignment();
  }

  const bool is_collective_kernel_enabled = absl::c_linear_search(
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_use_collective_kernels(),
      static_cast<int>(DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));

  absl::StatusOr<AllGatherInfo> maybe_info =
      BuildAllGatherInfo(is_collective_kernel_enabled, gpu_topology, all_gather,
                         device_assignment);
  if (!maybe_info.ok()) {
    VLOG(3) << "[CollectiveKernelStrategyAnnotator] Collective kernel not "
               "supported for AllGather "
            << instr->name() << ": " << maybe_info.status();
    return false;
  }

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr->backend_config<GpuBackendConfig>());
  gpu_config.mutable_collective_backend_config()->set_kernel_strategy(
      CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
  RETURN_IF_ERROR(instr->set_backend_config(gpu_config));

  VLOG(3) << "[CollectiveKernelStrategyAnnotator] Annotated AllGather "
          << instr->name() << " with KERNEL_STRATEGY_TRITON_ONE_SHOT";
  return true;
}

}  // namespace

CollectiveKernelStrategyAnnotator::CollectiveKernelStrategyAnnotator(
    const GpuTopology& gpu_topology, bool is_multimem_enabled)
    : gpu_topology_(gpu_topology), is_multimem_enabled_(is_multimem_enabled) {}

absl::StatusOr<bool> CollectiveKernelStrategyAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RET_CHECK(gpu_topology_.has_gpu_target_config())
      << "GpuTopology must have a target config for the strategy annotator.";
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce) {
        ASSIGN_OR_RETURN(
            bool annotated,
            TryAnnotateAllReduce(instr, gpu_topology_, is_multimem_enabled_));
        changed |= annotated;
      } else if (instr->opcode() == HloOpcode::kAllGather) {
        ASSIGN_OR_RETURN(bool annotated,
                         TryAnnotateAllGather(instr, gpu_topology_));
        changed |= annotated;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
