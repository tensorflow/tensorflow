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

#include "xla/backends/gpu/transforms/sycl_gemm_workspace_pass.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/sycl/sycl_matmul_utils.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<size_t> ComputeOneDnnScratchpadSize(
    const HloInstruction* matmul_instr,
    const se::GpuComputeCapability& gpu_version) {
  ABSL_ASSIGN_OR_RETURN(GemmConfig gemm_config,
                   GemmConfig::For(matmul_instr, gpu_version));
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   matmul_instr->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();
  ABSL_ASSIGN_OR_RETURN(auto prim_desc,
                   stream_executor::sycl::CreateMatMulPrimDescFromGemmConfig(
                       gemm_config, config.epilogue()));
  return prim_desc->scratchpad_desc().get_size();
}

}  // namespace

absl::StatusOr<bool> SyclGemmWorkspacePass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Resizes the trailing S8 workspace element of the matmul custom call's
  // tuple shape in place. As the workspace is consumed only by the custom
  // call at runtime, downstream get-tuple-element users require no update.
  auto resize_workspace = [](HloInstruction* matmul_instr, int64_t new_bytes) {
    Shape* shape = matmul_instr->mutable_shape();
    *shape->mutable_tuple_shapes(shape->tuple_shapes().size() - 1) =
        ShapeUtil::MakeShape(S8, {new_bytes});
  };

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* matmul_instr : computation->instructions()) {
      if (!IsCublasLtMatmul(*matmul_instr)) {
        continue;
      }
      const Shape& matmul_shape = matmul_instr->shape();
      if (!matmul_shape.IsTuple() || matmul_shape.tuple_shapes().size() != 2) {
        continue;
      }
      // oneDNN's matmul primitive does not support complex element types.
      if (primitive_util::IsComplexType(
              matmul_instr->shape().tuple_shapes(0).element_type())) {
        continue;
      }
      auto scratchpad_or =
          ComputeOneDnnScratchpadSize(matmul_instr, gpu_version_);
      if (!scratchpad_or.ok()) {
        VLOG(1) << "Failed to compute OneDNN scratchpad size for "
                << matmul_instr->custom_call_target() << ": "
                << scratchpad_or.status().message();
        continue;
      }
      resize_workspace(matmul_instr, static_cast<int64_t>(*scratchpad_or));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
