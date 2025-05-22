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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {
struct ClusterInfo;
}
}  // namespace triton
}  // namespace mlir

namespace xla {
namespace gpu {

struct TritonWrapperResult {
  int64_t shmem_bytes = 0;
  std::optional<se::ClusterDim> cluster_dim;
  std::optional<stream_executor::gpu::TmaMetadata> tma_metadata;

  // The captured nvvm.annotations from the lowest level LLVM IR coming from
  // Triton. We need to propagate them because we later create the kernel and
  // splice the impl_fn into it.
  std::vector<llvm::Metadata*> nvvm_annotations;
};

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
// If `emit_kernels` is false, then the function skips emitting
// the kernels, but it still returns correctly filled TritonWrapperResult.
// That is useful when deserializing from the compilation cache.
absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion,
    bool emit_kernel = true);

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info);

// TODO(b/406472229): Move the contents of this namespace to a helpers file
// to avoid polluting `fusion_emitter.h`.
// Exposed for testing purposes only. Do not use.
namespace ir_emitter_triton_internal {

// Computes the transformation from a 1-d program_id to a tile multi-index.
llvm::SmallVector<mlir::Value, 3> ComputeDelinearizedTileIndex(
    EmitterLocOpBuilder& b, absl::Span<const int64_t> num_output_tiles_per_dim);

// Dumps the Triton IR to a string.
//
// If `dump_annotations` is true, then the function also dumps the loc
// attributes of the instructions. Otherwise, it dumps the IR without
// annotations.
inline std::string DumpTritonIR(mlir::ModuleOp triton_module,
                                bool dump_annotations) {
  std::string triton_ir;
  llvm::raw_string_ostream os(triton_ir);
  triton_module.print(os, mlir::OpPrintingFlags().enableDebugInfo(
                              dump_annotations, dump_annotations));
  if (dump_annotations) {
    return EmitterLocOpBuilder::FormatTritonIrWithAnnotations(triton_ir);
  }
  return triton_ir;
}

}  // namespace ir_emitter_triton_internal
}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_
