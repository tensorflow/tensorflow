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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_EMITTER_BASE_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_EMITTER_BASE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class EmitterBase : public KernelFusionInterface {
 public:
  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

  // Visible for testing. `buffer_assignment` is optional for testing (assigns
  // a different buffer to each tensor).
  absl::StatusOr<std::unique_ptr<llvm::Module>> CreateLLVMModule(
      mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
      const se::DeviceDescription& device, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const;

  // Visible for testing. `buffer_assignment` is optional for testing (assigns
  // a different buffer to each tensor).
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const;

 protected:
  // Returns the set of instructions that will be isolated in the partitioned,
  // i.e., they will get their own subgraph. We won't automatically emit
  // functions for these instructions.
  virtual std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const {
    return {};
  }

  // Creates an epilogue with the raw thread/block/symbol indices, as defined
  // by the fusion's thread->output mapping.
  emitters::EpilogueSpecification GetEpilogueForOutputIndexing(
      const HloFusionAnalysis& analysis,
      const std::vector<const HloInstruction*>& heroes,
      const std::vector<const HloInstruction*>& roots,
      mlir::MLIRContext* mlir_context) const;

  virtual absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const = 0;

  // Evaluates the epilogue of the fusion. Returns the results for each epilogue
  // root.
  absl::flat_hash_map<const HloInstruction*, mlir::ValueRange> EmitEpilogue(
      int epilogue_index, const emitters::PartitionedComputations& computations,
      mlir::func::FuncOp entry_fn,
      const absl::flat_hash_map<const HloInstruction*,
                                llvm::SmallVector<mlir::Value>>& injected,
      mlir::ValueRange output_indices,
      mlir::ImplicitLocOpBuilder& builder) const;

  mlir::Value EmitWorkGroupId(mlir::ImplicitLocOpBuilder& builder,
                              WorkGroupDimension dim) const;
  mlir::Value EmitBlockId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  llvm::SmallVector<mlir::Value> EmitThreadAndBlockIds(
      mlir::ImplicitLocOpBuilder& builder) const;
  llvm::SmallVector<mlir::Value> EmitWorkGroupIds(
      mlir::ImplicitLocOpBuilder& builder) const;

 private:
  // Emits MLIR for the given fusion. The entry function has one tensor argument
  // per fusion parameter and output and one tensor result per fusion output.
  // The fuson outputs may only be used with `tensor.insert` ops.a
  absl::Status EmitMlir(mlir::ModuleOp module,
                        mlir::func::FuncOp entry_function,
                        const HloFusionInstruction& fusion) const;
};

// Adds passes that simplify arithmetic operations and remove dead code.
void AddXlaGpuOpsOptimizationPasses(mlir::OpPassManager& pm);

// Adds passes that transform XLA_GPU and SCF loops, e.g. peel, pipeline,
// vectorize.
void AddLoopTransformationPasses(mlir::OpPassManager& pm,
                                 const se::DeviceDescription& device);

// Adds passes that lower transformed loops to LLVM.
void AddLoweringPasses(mlir::OpPassManager& pm,
                       const se::DeviceDescription& device);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_EMITTER_BASE_H_
