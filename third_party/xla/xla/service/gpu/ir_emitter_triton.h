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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
#define XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/gemm_rewriter_triton.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

struct TritonWrapperResult {
  int64_t shmem_bytes;
};

// Compute the launch dimensions for the given Triton MatMul.
LaunchDimensions GetMatMulLaunchDimensions(
    const TritonFusionAnalysis& analysis,
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& fusion_boundary, const TritonGemmConfig& config);
// Use tiling and execution parameters from 'config'.
Status EmitMatMul(mlir::OpBuilder b, absl::string_view libdevice_path,
                  const TritonFusionAnalysis& analysis,
                  const HloComputation* computation, mlir::triton::FuncOp fn,
                  const TritonGemmConfig& config, int shmem_budget);

// Compute the launch dimensions for the given Triton SoftMax.
LaunchDimensions GetSoftMaxLaunchDimensions(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& fusion_boundary, const TritonGemmConfig& config);
// Generate Softmax in Triton IR inside 'fn'.
// Use execution parameters from 'config'.
Status EmitSoftMax(mlir::OpBuilder b, absl::string_view libdevice_path,
                   const TritonFusionAnalysis& analysis,
                   const HloComputation* computation, mlir::triton::FuncOp fn,
                   const TritonGemmConfig& config, int shmem_budget);

using TritonIrEmitter = std::function<Status(
    mlir::OpBuilder, absl::string_view, const TritonFusionAnalysis& analysis,
    const HloComputation*, mlir::triton::FuncOp, const TritonGemmConfig&, int)>;

// Generate Triton IR by running the provided generator and compile it into LLVM
// IR.
// MatMul and SoftMax above are some such IR generators.
StatusOr<TritonWrapperResult> TritonWrapper(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation, absl::string_view fusion_kind,
    const se::CudaComputeCapability& cc,
    const se::DeviceDescription& device_info, const TritonGemmConfig& config,
    llvm::Module* llvm_module, TritonIrEmitter ir_emitter,
    mlir::MLIRContext& mlir_context);

// Creates the initial Triton module for the given fusion. Visible for testing,
// use TritonWrapper instead.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation,
    const se::DeviceDescription& device_info, const TritonGemmConfig& config,
    TritonIrEmitter ir_emitter, mlir::MLIRContext& mlir_context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
