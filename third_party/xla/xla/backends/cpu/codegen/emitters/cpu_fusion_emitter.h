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
#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace cpu {

IndexingMap GetDefaultIndexingMap(absl::Span<const int64_t> thread_tile_sizes,
                                  absl::Span<const int64_t> shape,
                                  mlir::MLIRContext* mlir_context);

absl::StatusOr<mlir::func::FuncOp> EmitEntryFunctionApi(
    mlir::ModuleOp fusion_module, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment& buffer_assignment);

// Emit the call targets for the given fusion.
absl::StatusOr<emitters::CallTargetProvider> EmitCallTargets(
    mlir::ModuleOp module, const HloFusionInstruction& fusion,
    const emitters::PartitionedComputations& computations,
    const std::vector<emitters::EpilogueSpecification>& epilogues);

// Set the data layout attribute of the module based on the called instructions
// of the fusion.
void SetDataLayoutAttribute(mlir::ModuleOp module,
                            const HloFusionInstruction& fusion);

// Creates a module op with the name of the fusion using `GetFusionName`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateNamedMlirModuleOp(
    const HloFusionInstruction& fusion, mlir::Builder& builder);

// Returns the name of the fusion.
// If `xla_cpu_generate_unique_c_style_kernel_entry_points` is true, returns a
// C-style name of the fusion created by combining the name of the parent
// HloModule and the name of the fusion.
absl::StatusOr<std::string> GetFusionName(const HloFusionInstruction& fusion);

class CpuFusionEmitterBase {
 public:
  virtual ~CpuFusionEmitterBase() = default;

  virtual int64_t num_threads() const = 0;

  virtual std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const = 0;

  virtual std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t, int64_t, mlir::MLIRContext*) const = 0;

  virtual std::string BackendExtraOptions() { return {}; }

  virtual absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Emit() const = 0;
};

int64_t CeilDiv(int64_t a, int64_t b);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_H_
