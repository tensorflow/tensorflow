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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_COMPILER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/cost_model/block_level_parameters.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"

namespace mlir {
namespace triton {
}  // namespace triton
}  // namespace mlir

namespace xla {
namespace gpu {

struct TritonWrapperResult {
  int64_t shmem_bytes = 0;
  int64_t global_scratch_memory_size = 0;
  se::gpu::TmaMetadata tma_metadata;
  se::ThreadDim thread_dims;

  // The captured nvvm.annotations from the lowest level LLVM IR coming from
  // Triton. We need to propagate them because we later create the kernel and
  // splice the impl_fn into it.
  std::vector<llvm::Metadata*> nvvm_annotations;
  std::unique_ptr<llvm::Module> llvm_module;
};

std::ostream& operator<<(std::ostream& os, const TritonWrapperResult& result);

// Load the MLIR dialects required for Triton IR generation.
void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context);

// Generate Triton IR by running the provided generator and compile it into LLVM
// IR.
absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    llvm::LLVMContext& llvm_context, mlir::MLIRContext& mlir_context);

// Creates the initial Triton module for the given fusion. Visible for testing,
// use TritonWrapper instead.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context);

// Compiles a given Triton module to LLVM IR.
// If `emit_kernels` is false, then the function skips emitting
// the kernels, but it still returns correctly filled TritonWrapperResult.
// That is useful when deserializing from the compilation cache.
absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, const llvm::Triple& target_triple,
    const std::string& data_layout, llvm::LLVMContext& llvm_context,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion,
    bool emit_kernel = true);

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info);

// TODO(b/406472229): Move the contents of this namespace to a helpers file
// to avoid polluting `fusion_emitter.h`.
// Exposed for testing and experimental purposes only. Do not use.
namespace ir_emitter_triton_internal {

// Returns the MLIR module as a string.
inline std::string GetModuleIrString(mlir::ModuleOp triton_module,
                                     mlir::OpPrintingFlags flags = {}) {
  std::string triton_ir;
  llvm::raw_string_ostream os(triton_ir);
  triton_module.print(os, flags);
  return triton_ir;
}

// Given a tiling specification for a fusion and an annotated fusion, derives a
// tiling for the annotated fusion.
//
// Note that the tiling extracted here is voluntarily not checked against the
// specification, which means that it could be invalid. This should only be the
// case, though, if this logic gets stale, or if the fusion does not contain
// the required annotations. Checking constraints is not cheap, so we left it up
// to the caller to decide when to check the constraints.
//
// TODO(b/421837868): this belongs near/in `BlockLevelParameters`, but we start
// with this here in order to allow an incremental replacement.
absl::StatusOr<Tiling> TilingFromAnnotatedFusion(
    const HloFusionInstruction* fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters);

// This function lowers the shared dialect module to Triton. It is exposed for
// testing with the same motivation as EmitXTileModule.
//
// The `fusion` instruction should be the one that was used to create the shared
// dialect module.
absl::Status LowerXTileToTriton(
    mlir::ModuleOp xtile_dialect_module, mlir::MLIRContext& mlir_context,
    const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters);

}  // namespace ir_emitter_triton_internal
}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_COMPILER_H_
