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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace cpu {

struct CpuFusionEmissionResult {
  std::unique_ptr<llvm::Module> llvm_module;
  absl::flat_hash_set<int64_t> invariant_arguments;
};

IndexingMap GetDefaultIndexingMap(absl::Span<const int64_t> thread_tile_sizes,
                                  absl::Span<const int64_t> shape,
                                  mlir::MLIRContext* mlir_context);

class CpuFusionEmitterBase {
 public:
  CpuFusionEmitterBase(mlir::MLIRContext* mlir_context,
                       llvm::LLVMContext* llvm_context,
                       const BufferAssignment& buffer_assignment,
                       const HloFusionInstruction* fusion)
      : mlir_context_(mlir_context),
        llvm_context_(llvm_context),
        buffer_assignment_(buffer_assignment),
        fusion_(fusion) {}

  virtual ~CpuFusionEmitterBase() = default;

  virtual int64_t num_threads() const = 0;

  virtual std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const = 0;

  virtual std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t, int64_t, mlir::MLIRContext*) const = 0;

  virtual std::string BackendExtraOptions() { return {}; }

  absl::StatusOr<CpuFusionEmissionResult> Emit() const;

  // Visible for testing.
  absl::StatusOr<std::unique_ptr<llvm::Module>> CreateLLVMModule(
      mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
      const HloFusionInstruction& fusion,
      const BufferAssignment& buffer_assignment) const;

  // Visible for testing.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment& buffer_assignment,
      mlir::interpreter::MlirCompilationTrace* trace = nullptr) const;

 protected:
  virtual absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const = 0;

  virtual std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const {
    // We don't actually support epilogues for scatter, but this is how we tell
    // the base class that we don't want it to generate code for the scatter.
    return {};
  }

  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;

  mlir::MLIRContext* mlir_context_;
  llvm::LLVMContext* llvm_context_;
  const BufferAssignment& buffer_assignment_;
  const HloFusionInstruction* fusion_;

 private:
  // Emits MLIR for the given fusion. The entry function has one tensor argument
  // per fusion parameter and output and one tensor result per fusion output.
  // The fuson outputs may only be used with `tensor.insert` ops.a
  absl::Status EmitMlir(mlir::ModuleOp module,
                        mlir::func::FuncOp entry_function,
                        const HloFusionInstruction& fusion) const;
};

int64_t CeilDiv(int64_t a, int64_t b);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_H_
