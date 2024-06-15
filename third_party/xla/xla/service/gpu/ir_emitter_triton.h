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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;

struct TritonWrapperResult {
  int64_t shmem_bytes = 0;
  std::optional<se::ClusterDim> cluster_dim;
};

// Generate Triton IR inside 'fn'. This uses the given output_tile_sizes
// and the SymbolicTileAnalysis from the computation. The provided
// TritonFusionAnalysis and TritonGemmConfig are ignored.
absl::Status EmitGeneric(mlir::OpBuilder b, absl::string_view libdevice_path,
                         const se::DeviceDescription& device_info,
                         const HloFusionInstruction* fusion,
                         mlir::triton::FuncOp fn,
                         const BlockLevelParameters& block_level_parameters);

// Compute the launch dimensions for the given Triton MatMul.
absl::StatusOr<LaunchDimensions> GetMatMulLaunchDimensions(
    const TritonFusionAnalysis& analysis, const HloFusionAdaptor& fusion,
    const TritonGemmConfig& config);

// Use tiling and execution parameters from 'config'. output_tile_sizes is
// ignored.
absl::Status EmitMatMul(mlir::OpBuilder b, absl::string_view libdevice_path,
                        const se::DeviceDescription& device_info,
                        const HloFusionInstruction* fusion,
                        mlir::triton::FuncOp fn,
                        const BlockLevelParameters& block_level_parameters);

// Generate Softmax in Triton IR inside 'fn'.
// Use execution parameters from 'block_level_parameters'.
absl::Status EmitSoftMax(mlir::OpBuilder b, absl::string_view libdevice_path,
                         const se::DeviceDescription& device_info,
                         const HloFusionInstruction* fusion,
                         mlir::triton::FuncOp fn,
                         const BlockLevelParameters& block_level_parameters);

using TritonIrEmitter = std::function<absl::Status(
    mlir::OpBuilder, absl::string_view, const se::DeviceDescription&,
    const HloFusionInstruction*, mlir::triton::FuncOp,
    const BlockLevelParameters&)>;

// Load the MLIR dialects required for Triton IR generation.
void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context);

// Generate Triton IR by running the provided generator and compile it into LLVM
// IR.
absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    llvm::Module* llvm_module, mlir::MLIRContext& mlir_context);

// Creates the initial Triton module for the given fusion. Visible for testing,
// use TritonWrapper instead.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context);

// Compiles a given Triton module to LLVM IR.
absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    const HloModuleConfig& hlo_config, absl::string_view hlo_module_name,
    const se::GpuComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context);

// Create Triton pipeline.
//
// `out_cluster_info` must be kept alive at least until pm.run() is called.
// It should be read after that. We have to pass the cluster dims to
// LaunchDimensions. Triton currently uses this as an out-parameter to return
// the cluster dims determined based on `config.num_ctas` and a heuristic. There
// are some signs that show that this was intended to be used as an in-out
// parameter which would give a hint to Triton which cluster dims we prefer to
// use, but that's not the case currently.
absl::Status CreateTritonPipeline(
    mlir::OpPassManager& pm, const se::GpuComputeCapability& cc,
    const BlockLevelParameters& block_level_parameters,
    mt::nvidia_gpu::ClusterInfo& out_cluster_info);

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info);

// Exposed for testing purposes only. Do not use.
namespace ir_emitter_triton_internal {

// Used for creating Triton Load and Store ops.
struct MakeTensorPtrOpAndBoundaryChecks {
  mt::MakeTensorPtrOp op;

  // Indices of dimensions where the original tile size is not a power of 2 and
  // requires a boundary check.
  llvm::SmallVector<int32_t> boundary_checks;
};

MakeTensorPtrOpAndBoundaryChecks CreateMakeTensorPtrOp(
    mlir::ImplicitLocOpBuilder& b, mlir::Value pid,
    const TiledHloInstruction& tiled_hlo, mlir::Value argument_block);
}  // namespace ir_emitter_triton_internal

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_TRITON_H_
