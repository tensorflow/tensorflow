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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_COMPILATION_PIPELINE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_COMPILATION_PIPELINE_H_

#include <cstdint>

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu::tensor_ir {

// Returns the TensorIR-to-CudaTile pipeline options for the given target
// and tiling configuration. `tile_size` is the output tile shape and
// `reduction_tile_size` is the tile size for the contracting dimension (only
// relevant for fusions that contain a reduction).
mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions GetPipelineOptions(
    const stream_executor::GpuComputeCapability& gpu_cc,
    llvm::ArrayRef<int32_t> tile_size,
    int64_t reduction_tile_size =
        mlir::nv_tensor_ir::kDefaultReductionTileSize);

// Adds the TensorIR-to-CudaTile analysis and conversion pipeline to `pm`.
void CreateTensorIrPipeline(
    mlir::OpPassManager* pm,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions& options);

// Points TensorIR's Tile IR assembler at the `tileiras` binary staged in the
// runfiles. TensorIR only looks on `$PATH` by default, where a Bazel-staged
// binary never is, and quietly falls back to emitting Tile IR bytecode for the
// driver to JIT.
//
// Runs the lookup once and is safe to call from several threads. Missing an
// assembler is not an error: CUDA before 13.1 does not ship one, and the
// bytecode fallback still produces a working kernel wherever the driver can
// JIT it.
void SetUpTileIrAssembler();

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_COMPILATION_PIPELINE_H_
