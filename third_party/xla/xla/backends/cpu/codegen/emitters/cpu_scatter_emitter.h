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
#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_SCATTER_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_SCATTER_EMITTER_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace cpu {

// Maps (thread_id)[index_id, vector_id, vector_element_id] to indices of a
// simplified scatter updates operand.
IndexingMap GetScatterIndexingMap(
    absl::Span<const int64_t> updates_operand_shape, int64_t num_threads,
    int64_t vector_size, mlir::MLIRContext* context);

// Generic scatter fusion. Lowers to LLVM via MLIR.
class CpuScatterFusion final : public KernelEmitter<MlirKernelSource> {
 public:
  CpuScatterFusion(const BufferAssignment& buffer_assignment,
                   const HloFusionInstruction* fusion,
                   mlir::MLIRContext* mlir_context);

  absl::string_view name() const final { return "cpu_scatter_fusion"; }
  absl::StatusOr<KernelDefinition> EmitKernelDefinition() final;

 private:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const;

  const BufferAssignment& buffer_assignment_;
  const HloFusionInstruction* fusion_;
  mlir::MLIRContext* mlir_context_;

  int64_t num_threads_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_SCATTER_EMITTER_H_
