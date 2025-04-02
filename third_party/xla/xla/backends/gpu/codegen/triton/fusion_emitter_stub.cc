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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

absl::Status EmitGeneric(mlir::OpBuilder b, absl::string_view libdevice_path,
                         const se::DeviceDescription& device_info,
                         const HloFusionInstruction* fusion,
                         mlir::triton::FuncOp fn,
                         const BlockLevelParameters& block_level_parameters) {
  return absl::UnimplementedError("not supported for this build configuration");
}

void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context) {}

absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    llvm::Module* llvm_module, mlir::MLIRContext& mlir_context) {
  return absl::UnimplementedError("not supported for this build configuration");
}

absl::StatusOr<TritonModule> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context) {
  return absl::UnimplementedError("not supported for this build configuration");
}

absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion, bool emit_kernel) {
  return absl::UnimplementedError("not supported for this build configuration");
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  return "";
}

namespace ir_emitter_triton_internal {

llvm::SmallVector<mlir::Value, 3> ComputeDelinearizedTileIndex(
    EmitterLocOpBuilder& b,
    absl::Span<const int64_t> num_output_tiles_per_dim) {
  return {};
}

absl::StatusOr<mlir::triton::xla::TileOp> CreateTileOp(
    EmitterLocOpBuilder& b, mlir::ValueRange tile_multi_index,
    const TiledHloInstruction& tiled_hlo, mlir::Value parent_base_ptr) {
  return absl::UnimplementedError("not supported for this build configuration");
}
}  // namespace ir_emitter_triton_internal

}  // namespace gpu
}  // namespace xla
