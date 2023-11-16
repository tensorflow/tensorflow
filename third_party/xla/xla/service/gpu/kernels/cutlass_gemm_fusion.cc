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

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/kernels/custom_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm_kernel.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

class CutlassGemmFusion : public CustomFusion {
 public:
  StatusOr<std::vector<CustomKernel>> LoadKernels(
      const HloComputation* computation) const final {
    // TODO(ezhulenev): This is the most basic check to pass a single test we
    // have today. Expand it to properly check all invariants of a dot
    // instruction supported by CUTLASS gemm kernels.
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr)
      return absl::InternalError(
          "cutlass_gemm requires ROOT operation to be a dot");

    PrimitiveType dtype = dot->shape().element_type();
    if (dtype != PrimitiveType::F32)
      return absl::InternalError("Unsupported element type");

    auto& lhs_shape = dot->operand(0)->shape();
    auto& rhs_shape = dot->operand(1)->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    TF_ASSIGN_OR_RETURN(auto kernel,
                        kernel::GetCutlassGemmKernel(dtype, m, n, k));
    return std::vector<CustomKernel>{std::move(kernel)};
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm", ::xla::gpu::CutlassGemmFusion);
