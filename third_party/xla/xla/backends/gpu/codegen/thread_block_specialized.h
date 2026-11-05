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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
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

// Emits a thread-block-specialized wrapper kernel of nested fusions.
class ThreadBlockSpecializedFusion : public KernelFusionInterface {
 public:
  struct LlvmKernelSourceWrapper {
    LlvmKernelSource source;
    int64_t max_shmem_bytes;
  };

  explicit ThreadBlockSpecializedFusion(const HloFusionAnalysis& analysis);

  static std::unique_ptr<MlirKernelEmitter> CreateNativeNestedEmitter(
      const HloFusionAnalysis& nested_analysis);

  LaunchDimensions launch_dimensions() const override {
    return launch_dimensions_;
  }
  std::optional<se::ClusterDim> cluster_dim() const { return cluster_dim_; }

  AsyncThunkSequence Emit(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion) const override;

  // Emits a launcher function that calls nested kernels based on the block IDs.
  // Each nested fusion is lowered separately with corresponding emitters,
  // virtualized and spliced into the main launcher module.
  // Result module is lowered to `llvm` + `nvvm` dialects.
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
  struct NestedKernelSpec {
    static NestedKernelSpec Create(const HloFusionInstruction& fusion,
                                   const se::DeviceDescription& device_info);

    const HloFusionInstruction* fusion = nullptr;
    LaunchDimensions launch_dims;
    std::vector<int64_t> outer_root_indices;
    HloFusionAnalysis::EmitterFusionKind emitter_fusion_kind;
    se::ClusterDim cluster_dim{1, 1, 1};
  };

  // Emits the entry function that conditionally launches nested kernels based
  // on block and thread IDs.
  absl::Status EmitLauncherBody(
      mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const;

  const se::DeviceDescription& device_info_;
  std::vector<NestedKernelSpec> nested_kernels_;
  LaunchDimensions launch_dimensions_;
  std::optional<se::ClusterDim> cluster_dim_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_THREAD_BLOCK_SPECIALIZED_H_
