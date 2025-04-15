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
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace cpu {

// Generic scatter fusion. Lowers to LLVM via MLIR.
class CpuScatterFusion : public CpuFusionEmitterBase {
 public:
  explicit CpuScatterFusion(mlir::MLIRContext* mlir_context,
                            llvm::LLVMContext* llvm_context,
                            const BufferAssignment& buffer_assignment,
                            const HloFusionInstruction* fusion);

  int64_t num_threads() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

  std::string BackendExtraOptions() override;

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Emit() const final;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const;

 private:
  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;

  mlir::MLIRContext* mlir_context_;
  llvm::LLVMContext* llvm_context_;
  const BufferAssignment& buffer_assignment_;
  const HloFusionInstruction* fusion_;

  int64_t vector_size_;
  int64_t num_threads_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_SCATTER_EMITTER_H_
