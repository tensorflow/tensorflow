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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_COMPILATION_PIPELINE_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_COMPILATION_PIPELINE_H_

#include <string>

#include "absl/status/status.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::triton::nvidia_gpu {

// Forward declaration to avoid including a GPU-only header.
struct ClusterInfo;

}  // namespace mlir::triton::nvidia_gpu

namespace xla {
namespace gpu {

// Creates a Triton compilation pipeline.
//
// `out_cluster_info` must be kept alive at least until pm.run() is called.
// It should be read after that. We have to pass the cluster dims to
// LaunchDimensions. Triton currently uses this as an out-parameter to return
// the cluster dims determined based on `config.num_ctas` and a heuristic. There
// are some signs that show that this was intended to be used as an in-out
// parameter which would give a hint to Triton which cluster dims we prefer to
// use, but that's not the case currently.
absl::Status CreateTritonPipeline(
    mlir::OpPassManager* pm, std::string arch_name, int num_warps, int num_ctas,
    int num_stages, mlir::triton::nvidia_gpu::ClusterInfo& out_cluster_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_COMPILATION_PIPELINE_H_
