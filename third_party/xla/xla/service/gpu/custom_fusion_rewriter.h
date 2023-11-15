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

#ifndef XLA_SERVICE_GPU_CUSTOM_FUSION_REWRITER_H_
#define XLA_SERVICE_GPU_CUSTOM_FUSION_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/kernels/custom_fusion_pattern.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla::gpu {

// Pattern matches HLO instruction to custom fusions (hand written CUDA C++
// kernels, e.g. custom GEMMs implemented with CUTLASS) and rewrites them into
// fusion instructions and fusion computations.
//
// Example: pattern matching dot operation into CUTLASS gemm
//
//  ENTRY %main (p0: f16[15,19], p1: f16[19,17]) -> f16[15,17] {
//    %p0 = f16[15,19]{1,0} parameter(0)
//    %p1 = f16[19,17]{1,0} parameter(1)
//    ROOT %r = f16[15,17]{1,0} dot(%p0, %p1),
//      lhs_contracting_dims={1}, rhs_contracting_dims={0}
//  }
//
// After the pass:
//
//  %cutlass_gemm (p0: f16[19,17], p1: f16[15,19]) -> f16[15,17] {
//    %p0 = f16[15,19]{1,0} parameter(0)
//    %p1 = f16[19,17]{1,0} parameter(1)
//    ROOT %r = f16[15,17]{1,0} dot(%p0, %p1),
//      lhs_contracting_dims={1}, rhs_contracting_dims={0}
//  }
//
//  ENTRY %main (p0: f16[15,19], p1: f16[19,17]) -> f16[15,17] {
//    %p0 = f16[15,19]{1,0} parameter(0)
//    %p1 = f16[19,17]{1,0} parameter(1)
//    ROOT %r = f16[15,17]{1,0} fusion(%p0, %p1), kind=kCustom,
//      calls==cutlass_gemm,
//      backend_config={kind: "__custom_fusion",
//                      custom_fusion_config: {"name":"cutlass_gemm"}}
//  }
//
class CustomFusionRewriter : public HloModulePass {
 public:
  explicit CustomFusionRewriter(const CustomFusionPatternRegistry* patterns =
                                    CustomFusionPatternRegistry::Default());

  absl::string_view name() const override { return "custom-fusion-rewriter"; }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const CustomFusionPatternRegistry* patterns_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_CUSTOM_FUSION_REWRITER_H_
