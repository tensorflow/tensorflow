/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <functional>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

struct TritonWrapperResult {
  int64_t shmem_bytes = 0;
  std::optional<se::ClusterDim> cluster_dim;
};

// Compute the launch dimensions for the given Triton MatMul.
LaunchDimensions GetMatMulLaunchDimensions(const TritonFusionAnalysis& analysis,
                                           const HloFusionAdaptor& fusion,
                                           const TritonGemmConfig& config);
// Use tiling and execution parameters from 'config'.
absl::Status EmitMatMul(mlir::OpBuilder b, absl::string_view libdevice_path,
                        const se::DeviceDescription& device_info,
                        const TritonFusionAnalysis& analysis,
                        const HloComputation* computation,
                        mlir::triton::FuncOp fn,
                        const TritonGemmConfig& config);

// Compute the launch dimensions for the given Triton SoftMax.
LaunchDimensions GetSoftMaxLaunchDimensions(const HloFusionAdaptor& fusion,
                                            const TritonGemmConfig& config);
// Generate Softmax in Triton IR inside 'fn'.
// Use execution parameters from 'config'.
absl::Status EmitSoftMax(mlir::OpBuilder b, absl::string_view libdevice_path,
                         const se::DeviceDescription& device_info,
                         const TritonFusionAnalysis& analysis,
                         const HloComputation* computation,
                         mlir::triton::FuncOp fn,
                         const TritonGemmConfig& config);

using TritonIrEmitter = std::function<Status(
    mlir::OpBuilder, absl::string_view, const se::DeviceDescription&,
    const TritonFusionAnalysis& analysis, const HloComputation*,
    mlir::triton::FuncOp, const TritonGemmConfig&)>;

// Generate Triton IR by running the provided generator and compile it into LLVM
// IR.
// MatMul and SoftMax above are some such IR generators.
absl::StatusOr<TritonWrapperResult> TritonWrapper(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation, absl::string_view fusion_kind,
    const se::CudaComputeCapability& cc,
    const se::DeviceDescription& device_info, const TritonGemmConfig& config,
    llvm::Module* llvm_module, TritonIrEmitter ir_emitter,
    mlir::MLIRContext& mlir_context);

// Creates the initial Triton module for the given fusion. Visible for testing,
// use TritonWrapper instead.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation,
    const se::DeviceDescription& device_info, const TritonGemmConfig& config,
    TritonIrEmitter ir_emitter, mlir::MLIRContext& mlir_context);

// Compiles a given Triton module to LLVM IR.
absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    const HloModuleConfig& hlo_config, absl::string_view hlo_module_name,
    const se::CudaComputeCapability& cc,
    const se::DeviceDescription& device_info, const TritonGemmConfig& config,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
