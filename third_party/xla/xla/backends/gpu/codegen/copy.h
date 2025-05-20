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
#ifndef XLA_BACKENDS_GPU_CODEGEN_COPY_H_
#define XLA_BACKENDS_GPU_CODEGEN_COPY_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

// Special case of a fusion consisting only of `kCopy` instructions that can be
// implemented using `memcpy`s.
class MemcpyFusion : public FusionInterface {
 public:
  MemcpyFusion(const HloFusionAnalysis& analysis,
               const BufferAssignment* buffer_assignment)
      : analysis_(analysis), buffer_assignment_(buffer_assignment) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

 private:
  const HloFusionAnalysis& analysis_;
  const BufferAssignment* buffer_assignment_;
};

// Special case of a fusion consisting only of instructions that can be
// implemented using `memcpy`s. The difference between this fusion and
// `MemcpyFusion` is that here we allow `memcpy`s that have dynamic offsets
// (e.g. dynamic-slice in a while loop).
class DynamicMemcpyFusion : public FusionInterface {
 public:
  DynamicMemcpyFusion(const HloFusionAnalysis& analysis,
                      const BufferAssignment* buffer_assignment)
      : analysis_(analysis), buffer_assignment_(buffer_assignment) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

  // Inexpensive checks to see if a fusion might be a dynamic memcpy fusion.
  // If this returns true, GetMemcpyDescriptorForFusion might still fail.
  static bool IsCandidateFusion(const HloFusionInstruction& fusion);

  // Attempts to build a memcpy descriptor for the given fusion.
  static std::optional<DynamicMemcpyThunk::MemcpyDescriptor>
  GetMemcpyDescriptorForFusion(const HloFusionInstruction& fusion);

 private:
  const HloFusionAnalysis& analysis_;
  const BufferAssignment* buffer_assignment_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_COPY_H_
