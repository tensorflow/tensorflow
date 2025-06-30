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

#include <string>

#include "absl/status/status.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

namespace xla {
namespace gpu {

absl::Status CreateTritonPipeline(
    mlir::OpPassManager* pm, std::string arch_name, int num_warps, int num_ctas,
    int num_stages, mlir::triton::nvidia_gpu::ClusterInfo& out_cluster_info) {
  return absl::UnimplementedError("not supported for this build configuration");
}

}  // namespace gpu
}  // namespace xla
