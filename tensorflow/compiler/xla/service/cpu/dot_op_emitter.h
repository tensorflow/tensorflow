/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_H_

#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace cpu {
// Returns true if the two operands and the output of `dot_instr` must have row
// major layout.
bool DotOperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features);

// Returns true our lowering strategy for `dot_instr` can fold in transposes to
// the either of the inputs.
bool DotImplementationCanHandleTranspose(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features);

// Returns the index for an operand to `hlo` that should ideally be column
// major.  Returns nullopt if there is no such operand or if `hlo` is not a dot
// or a fusion containing a dot.
absl::optional<int64_t> ProfitableToMakeDotOperandColumnMajor(
    const HloInstruction& hlo);

// Emit LLVM IR to perform the dot operation on lhs_array and rhs_array and
// place the result in target_array. IR is emitted at current insert point of
// the builder. Upon completion of the method, the insert point is set to the
// end of all instructions emitted for this operation.
//
// If `addend_array` is not nullptr then it must be an array of the same
// dimensions as the result, and the result is computed as `addend_array` +
// dot(`lhs_array`, `rhs_array`).  A non-null `addend_array` is only supported
// for Matrix-vector products.
Status EmitDotOperation(const HloInstruction& dot,
                        const llvm_ir::IrArray& target_array,
                        const llvm_ir::IrArray& lhs_array,
                        const llvm_ir::IrArray& rhs_array,
                        const llvm_ir::IrArray* addend_array,
                        llvm::Value* executable_run_options_value,
                        llvm::IRBuilder<>* b, mlir::MLIRContext* mlir_context,
                        const HloModuleConfig& hlo_module_config,
                        const TargetMachineFeatures& target_machine_features);
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_H_
