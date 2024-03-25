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

#include "xla/service/gpu/fusions/cudnn.h"

#include "absl/strings/escaping.h"
#include "xla/hlo/ir/hlo_instructions.h"
#if GOOGLE_CUDA
#include "xla/service/gpu/runtime/cudnn_thunk.h"
#endif

namespace xla {
namespace gpu {

absl::StatusOr<FusionEmissionResult> CuDnnFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
#if GOOGLE_CUDA
  VLOG(3) << fusion.ToString();

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  if (!gpu_config.fusion_backend_config().has_cudnn_fusion_config() ||
      !gpu_config.fusion_backend_config()
           .cudnn_fusion_config()
           .has_serialized_graph()) {
    return absl::FailedPreconditionError("cuDNN fusion is not compiled.");
  }
  std::string unescaped;
  std::string error;
  if (!absl::CUnescape(gpu_config.fusion_backend_config()
                           .cudnn_fusion_config()
                           .serialized_graph(),
                       &unescaped, &error)) {
    return absl::UnknownError(
        absl::StrCat("Failed unescaping string: ", error));
  }

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));
  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<CuDnnThunk>(
      std::move(unescaped), Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
      kernel_arguments.args()));
  return result;
#else
  return absl::UnimplementedError("cuDNN support requires CUDA");
#endif
}

}  // namespace gpu
}  // namespace xla
