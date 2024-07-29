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
#ifndef XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_
#define XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

// A wrapper for fusions implemented using the mechanism in
// xla/service/gpu/kernels. See custom_kernel_fusion.h in that folder for
// details.
class CustomFusion : public FusionInterface {
 public:
  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;
};

// Emitter for custom fusions implementing address computation. An address
// computation contains a custom call hero, with at least one of its operands
// coming from a static contiguous slice. E.g. operand `%cast` of `%gemm` coming
// from `%slice`:
// %address_computation {
//   %p0 = f32[2, 1024, 1024]
//   %p1 = f32[1024, 1024]
//   %slice = f32[1, 1024, 1024] slice(%p0)
//   %cast = f32[1024, 1024] bitcast(%slice)
//   ROOT %gemm = custom_call(%cast, %p1) __cublas$Gemm
// }
//
// The goal is to compute the buffer addresses for such operands (`%cast`) at
// compile-time instead of allocating a new buffer for it at runtime by
// translating the static slice into offset + size of the original buffer passed
// into the custom call `%gemm`.
class DynamicSliceFusion : public FusionInterface {
 public:
  explicit DynamicSliceFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

 private:
  const HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_
