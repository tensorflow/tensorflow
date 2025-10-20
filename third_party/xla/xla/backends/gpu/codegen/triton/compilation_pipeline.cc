/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include <variant>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla::gpu {

void CreateTritonXlaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, bool rewrite_int4,
    bool allow_tma) {
  pm->addPass(mlir::triton::xla::CreateTritonXLASqueezeDimsPass());
  pm->addPass(mlir::triton::xla::CreateTritonXLAFoldTransposePass());

  auto* cuda_cc = gpu_cc.cuda_compute_capability();
  bool is_at_least_hopper = cuda_cc != nullptr && cuda_cc->IsAtLeastHopper();

  if (rewrite_int4) {
    pm->addPass(mlir::triton::xla::CreateInt4ToPackedInt4RewritePass(
        /*enable_bf16x2=*/is_at_least_hopper));
  }

  pm->addPass(mlir::triton::xla::CreateTritonXLAExtractInsertToTritonPass(
      /*allow_tma=*/allow_tma && is_at_least_hopper));

  // Lower affine expressions into arithmetic ops.
  pm->addPass(mlir::createLowerAffinePass());

  // Lower xla_gpu.apply_indexing into arithmetic ops.
  pm->addPass(emitters::CreateSimplifyAffinePass());
  pm->addPass(CreateConvertIndexTypePass());
  // We need LICM before unswitching loops because loop unswitcher relies on
  // having loop invariant code to be outside of the loop.
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::triton::xla::CreateTritonXLAUnswitchLoopsPass());
}

void CreateTritonCudaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::CudaComputeCapability& cuda_cc, int num_warps,
    int num_ctas, int num_stages,
    mlir::triton::nvidia_gpu::ClusterInfo& out_cluster_info);

void CreateTritonRocmPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::RocmComputeCapability& rocm_cc, int num_warps,
    int num_ctas, int num_stages);

void CreateTritonPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, int num_warps,
    int num_ctas, int num_stages,
    mlir::triton::nvidia_gpu::ClusterInfo& out_cluster_info) {
  if (auto* cuda_cc = gpu_cc.cuda_compute_capability()) {
    return CreateTritonCudaPipeline(pm, *cuda_cc, num_warps, num_ctas,
                                    num_stages, out_cluster_info);
  }

  CreateTritonRocmPipeline(pm, *gpu_cc.rocm_compute_capability(), num_warps,
                           num_ctas, num_stages);
  // There is no clusters in ROCm for now.
  out_cluster_info.clusterDimX = 1;
  out_cluster_info.clusterDimY = 1;
  out_cluster_info.clusterDimZ = 1;
}

}  // namespace xla::gpu
