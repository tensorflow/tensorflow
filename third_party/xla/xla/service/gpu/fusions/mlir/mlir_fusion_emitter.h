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

#include <memory>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/model/indexing_map.h"

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
  // Emits MLIR for the given fusion. The entry function has one tensor argument
  // per fusion parameter and output and one tensor result per fusion output.
  // The fuson outputs may only be used with `tensor.insert` ops.a
  virtual absl::Status EmitMlir(mlir::ModuleOp module,
                                mlir::func::FuncOp entry_function,
                                const HloFusionInstruction& fusion) const = 0;

  // Emit a loop nest for the symbols in the output map. The output map should
  // have the dimensions specified in KernelFusionInterface. Loops are nested
  // with the symbol 0 as the outermost loop. `output_indices` are the final
  // output indices, not just the indices of the symbols. The return value of
  // the function is the updated output tensors.
  absl::StatusOr<llvm::SmallVector<mlir::Value>> EmitLoopNest(
      mlir::ImplicitLocOpBuilder& b, mlir::ValueRange output_tensors,
      const IndexingMap& thread_to_output_map,
      const std::function<absl::StatusOr<llvm::SmallVector<mlir::Value>>(
          mlir::ValueRange output_tensors, mlir::ValueRange output_indices)>&
          create_body) const;

  mlir::Value EmitBlockId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_MLIR_FUSION_EMITTER_H_
