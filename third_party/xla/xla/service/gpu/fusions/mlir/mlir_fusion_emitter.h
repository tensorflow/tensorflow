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
#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_MLIR_FUSION_EMITTER_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_MLIR_FUSION_EMITTER_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class MlirFusionEmitterBase : public KernelFusionInterface {
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
  virtual absl::flat_hash_set<const HloInstruction*>
  GetInstructionsWithCustomCodegen(const HloFusionInstruction& fusion) const {
    return {};
  }

  virtual absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const = 0;

  // If the root is not the same as the hero, emits the epilogue for the hero.
  // The hero must have been passed in `GetInstructionsWithCustomCodegen`.
  mlir::ValueRange EmitEpilogue(
      const HloInstruction* root, const HloInstruction* hero,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::ValueRange injected_values, mlir::ValueRange output_indices,
      mlir::ImplicitLocOpBuilder& builder) const;

  // Emit a loop nest for the symbols in the output map. The map should have
  // the dimensions specified in KernelFusionInterface. Loops are nested with
  // the symbol 0 as the outermost loop. The indices of the map's dimensions and
  // symbols are passed to the lambda separately. The return values of the
  // function are the updated outputs.
  llvm::SmallVector<mlir::Value> EmitThreadLoopNest(
      mlir::ImplicitLocOpBuilder& b, mlir::ValueRange outputs,
      const IndexingMap& indexing_map,
      const std::function<llvm::SmallVector<mlir::Value>(
          mlir::ValueRange outputs, mlir::ValueRange dim_values,
          mlir::ValueRange symbol_values)>& create_body) const;

  mlir::Value EmitBlockId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;

 private:
  // Emits MLIR for the given fusion. The entry function has one tensor argument
  // per fusion parameter and output and one tensor result per fusion output.
  // The fuson outputs may only be used with `tensor.insert` ops.a
  absl::Status EmitMlir(mlir::ModuleOp module,
                        mlir::func::FuncOp entry_function,
                        const HloFusionInstruction& fusion) const;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_MLIR_FUSION_EMITTER_H_
