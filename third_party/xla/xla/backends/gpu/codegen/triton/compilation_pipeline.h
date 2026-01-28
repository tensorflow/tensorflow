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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_

#include "mlir/Pass/PassManager.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Adds TritonXLA passes to the pipeline.
void CreateTritonXlaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, bool rewrite_int4,
    bool allow_tma, int num_stages, bool warp_specialization_allowed);

// Creates a Triton compilation pipeline.
void CreateTritonPipeline(mlir::OpPassManager* pm,
                          const stream_executor::GpuComputeCapability& gpu_cc,
                          int num_warps, int num_ctas, int num_stages);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_COMPILATION_PIPELINE_H_
