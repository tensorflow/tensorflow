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
#ifndef XLA_BACKENDS_GPU_CODEGEN_PHASED_BLOCK_CLUSTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_PHASED_BLOCK_CLUSTER_H_

#include <cstddef>
#include <cstdint>
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
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

// Emits a single thread-block-cluster kernel (Hopper SM90+) executing nested
// fusions in sequential phases. Phases are separated by thread-block-cluster
// barriers and exchange intermediate results via DRAM or caches.
//
// This is an optimization for dependent small fusions whose execution would
// otherwise be dominated by fixed kernel launch and scheduling overheads.
// Executing them as phases of a single kernel pays that overhead only once.
//
// An outer `kCustom` wrapper fusion (`kind == "__phased_block_cluster"`)
// encapsulates a DAG of nested fusions returned via `ROOT tuple(...)`. Each
// nested fusion is lowered by its native emitter, virtualized
// (`VirtualizeSubKernelPass`), and spliced into a launcher kernel configured as
// a single thread-block cluster (`nvvm.cluster_dim`). Independent fusions
// within a phase execute concurrently, while phases synchronize via aligned
// `nvvm.cluster.arrive` / `nvvm.cluster.wait` barriers outside all CTA/thread
// guards. Intermediate results are written to the outer `ROOT` tuple buffers
// (with `llvm.noalias` stripped so LLVM preserves ordering across barriers) and
// read directly by consumer phases.
//
// Current limitations (to be lifted in follow-up CLs):
//   * Only straight-line chains of single-array `kLoop` fusions ordered
//     topologically in `ROOT tuple(f_0, ..., f_{N-1})` (no general DAGs or
//     non-`kLoop` nested fusions yet).
//   * Each nested fusion must use a 1D grid fitting in one portable cluster
//     (`<= 8` blocks; multi-wave block loops are not yet supported).
//   * No buffer aliasing with fusion outputs (`GpuAliasInfo` and explicit
//     `output_to_operand_aliasing` are not yet supported).
class PhasedBlockClusterFusion : public KernelFusionInterface {
 public:
  static constexpr uint64_t kMaxBlocksPerCluster = 8;

  explicit PhasedBlockClusterFusion(
      const HloFusionAnalysis& analysis,
      std::optional<LaunchDimensions> outer_launch_dims = std::nullopt);

  LaunchDimensions launch_dimensions() const override {
    return launch_dimensions_;
  }
  se::ClusterDim cluster_dim() const { return cluster_dim_; }

  AsyncThunkSequence Emit(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion) const override;

  // Visible for testing. Returns the spliced launcher module in `llvm`/`nvvm`.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateAndLowerMLIRModule(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      absl::string_view entry_function_name,
      const BufferAssignment* buffer_assignment) const;

 private:
  struct NestedKernelSpec {
    const HloFusionInstruction* fusion = nullptr;
    LaunchDimensions launch_dims;
  };

  absl::Status EmitLauncherBody(
      mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const;

  const se::DeviceDescription& device_info_;
  std::vector<NestedKernelSpec> nested_kernels_;
  LaunchDimensions launch_dimensions_;
  se::ClusterDim cluster_dim_{1, 1, 1};
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_PHASED_BLOCK_CLUSTER_H_
