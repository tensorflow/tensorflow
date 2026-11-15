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
#ifndef XLA_BACKENDS_GPU_CODEGEN_THREAD_BLOCK_SPECIALIZED_H_
#define XLA_BACKENDS_GPU_CODEGEN_THREAD_BLOCK_SPECIALIZED_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

// Emits a single kernel for arbitrarily nested horizontal
// (`__thread_block_specialized`) and phased (`__phased_block_cluster`) fusions.
//
// Each leaf fusion (`kLoop`, `kReduction`, `kTranspose`, `kScatter`,
// `kConcatenate`, `kTriton`) is lowered by its native emitter, virtualized once
// via `VirtualizeSubKernelPass`, and spliced as an `alwaysinline` internal
// function into a unified `@launcher` kernel. Horizontal nodes partition
// `block_id` across children, while phased nodes sequence children with block
// barriers between consecutive single-block phases and cluster barriers across
// multi-block boundaries.
class ThreadBlockSpecializedFusion : public KernelFusionInterface {
 public:
  static constexpr uint64_t kMaxBlocksPerCluster = 8;

  struct LlvmKernelSourceWrapper {
    LlvmKernelSource source;
    int64_t max_shmem_bytes;
  };

  struct NestedKernelSpec {
    enum class Kind { kLeaf, kHorizontal, kPhased };
    Kind kind = Kind::kLeaf;
    const HloFusionInstruction* fusion = nullptr;
    LaunchDimensions launch_dims;
    std::optional<se::ClusterDim> cluster_dim;
    // Whether small leaf fusions should be unrolled/vectorized.
    bool unroll_small_fusions = false;
    std::vector<int64_t> input_arg_indices;
    std::vector<int64_t> output_arg_indices;
    std::vector<NestedKernelSpec> children;
  };

  explicit ThreadBlockSpecializedFusion(const HloFusionAnalysis& analysis);

  static std::unique_ptr<MlirKernelEmitter> CreateNativeNestedEmitter(
      const HloFusionAnalysis& nested_analysis, bool unroll_small_fusions);

  LaunchDimensions launch_dimensions() const override {
    return root_spec_.launch_dims;
  }
  std::optional<se::ClusterDim> cluster_dim() const {
    return root_spec_.cluster_dim;
  }

  AsyncThunkSequence Emit(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion) const override;

  // Emits a launcher function that dispatches nested leaf kernels according to
  // the recursive `NestedKernelSpec` tree. Each leaf fusion is lowered
  // separately by its native emitter, virtualized, and spliced into the main
  // launcher module in `llvm` + `nvvm` dialects.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateAndLowerMLIRModule(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      absl::string_view entry_function_name,
      const BufferAssignment* buffer_assignment) const;

  // Visible for testing.
  absl::StatusOr<LlvmKernelSourceWrapper> CreateLLVMModule(
      const se::DeviceDescription& device, const HloFusionInstruction& fusion,
      absl::string_view entry_function_name,
      const BufferAssignment* buffer_assignment,
      mlir::MLIRContext& mlir_context) const;

 private:
  // Emits the entry function that conditionally launches nested kernels based
  // on block and thread IDs.
  absl::Status EmitLauncherBody(
      mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      absl::Span<const NestedKernelSpec* const> leaf_specs,
      absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const;

  const se::DeviceDescription& device_info_;
  NestedKernelSpec root_spec_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_THREAD_BLOCK_SPECIALIZED_H_
