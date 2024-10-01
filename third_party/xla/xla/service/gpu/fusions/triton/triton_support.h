/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_TRITON_SUPPORT_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_TRITON_SUPPORT_H_

// This file is the home of the basic Triton support checks which are used by
// multiple other components.

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using CodegenDecision = FusionDecision;

// Checks that Triton officially supports the provided compute capability.
//
// Currently does not perform any check for non-CUDA compute capabilities.
absl::Status EnsureTritonSupportsComputeCapability(
    const se::GpuComputeCapability& gpu_compute_capability);

// Return `CodegenDecision`'s equivalent of `true` if the parameter instruction
// is supported by the Triton emitters for the given compute capability. Note
// that this function makes no assumption about what happens if
// `FloatNormalization` is run, unlike the legacy Triton utils.
//
// Note: this function is entirely dissociated from the legacy Triton emitters.
// If you intend to add a feature to the legacy Triton emitters (which you
// probably shouldn't), use `legacy_triton::IsTritonSupportedInstruction`
// instead.
CodegenDecision IsTritonSupportedInstruction(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version);

// Returns `true` if the parameter computation is a Triton fused computation,
// i.e. the calling fusion instruction has `FusionKind::kCustom` and
// `backend_config<gpu::GpuBackendConfig>()` with `kind` set to
// `kTritonGemmFusionKind`.
bool IsTritonFusedComputation(const HloComputation& computation);

namespace internal {
// TODO(b/363981282): Remove the function below once all ops are tested via
// HLOs. This is exposed for testing purposes only and will be removed in the
// near future. Do not use. This functions only returns a partial result.
bool IsTritonUnsupportedOpcode(HloOpcode opcode);
}  // namespace internal

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_TRITON_SUPPORT_H_
