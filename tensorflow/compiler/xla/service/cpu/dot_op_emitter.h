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
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

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
absl::optional<int64> ProfitableToMakeDotOperandColumnMajor(
    const HloInstruction& hlo);

// Helper class for emitting LLVM IR to perform the dot operation.
class DotOpEmitter {
 public:
  // Emit LLVM IR to perform the dot operation on lhs_array and rhs_array and
  // place the result in target_array. IR is emitted at current insert point of
  // the builder. Upon completion of the method, the insert point is set to the
  // end of all instructions emitted for this operation.
  //
  // If `addend_array` is not nullptr then it must be an array of the same
  // dimensions as the result, and the result is computed as `addend_array` +
  // dot(`lhs_array`, `rhs_array`).  A non-null `addend_array` is only supported
  // for Matrix-vector products.
  static Status EmitDotOperation(
      const HloInstruction& dot, const llvm_ir::IrArray& target_array,
      const llvm_ir::IrArray& lhs_array, const llvm_ir::IrArray& rhs_array,
      const llvm_ir::IrArray* addend_array,
      llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
      const HloModuleConfig& hlo_module_config,
      const TargetMachineFeatures& target_machine_features);

 private:
  DotOpEmitter(const HloInstruction& dot, const llvm_ir::IrArray& target_array,
               const llvm_ir::IrArray& lhs_array,
               const llvm_ir::IrArray& rhs_array,
               const llvm_ir::IrArray* addend_array,
               llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
               const HloModuleConfig& hlo_module_config,
               const TargetMachineFeatures& target_machine_features);

  // Emits the IR to perform the dot operation.
  Status Emit();

  // Emits instructions to perform a scalar dot product (a multiply of the
  // LHS and RHS) and store the results in the target.
  Status EmitScalarDot();

  // Emits a call to the CPU runtime to perform the matrix multiply.
  Status EmitCallToRuntime();

  // Represents the dimensions of a matrix-matrix multiply operation.
  struct MatMultDims {
    // The number of rows in the LHS.
    int64 m;

    // The number of columns in the LHS, which is also must be equal to the
    // number of rows in the RHS.
    int64 k;

    // The number of columns on the RHS.
    int64 n;

    // True if the LHS matrix is column major.
    bool lhs_column_major;

    // True if the LHS contraction dimension is not 1.
    bool lhs_non_canonical;

    // True if the RHS matrix is column major.
    bool rhs_column_major;

    // True if the RHS contraction dimension is not 0.
    bool rhs_non_canonical;

    // True if the result matrix is column major.
    bool target_column_major;
  };

  // Get the MatMultDims instance for the dot product this DotOpEmitter
  // represents.  Precondition: the dot is of rank 2 (and thus its operands are
  // of rank 2 as well).
  MatMultDims GetMatMultDims() const;

  // Lowers the dot operation as a tiled Matrix*Vector loop.
  void EmitTiledLlvmIrGemv();

  // Lowers the dot operation as a tiled Matrix*Matrix loop.
  void EmitTiledLlvmIrGemm();

  // Lowers the dot operation as a naive nested loop that computes the result
  // one element at a time.
  void EmitNaiveLlvmIrGemm();

  // When doing a tiled GEMV in LLVM IR, a "tile" consists of this many vector
  // registers.
  int64 GetGemvTilingFactor() const {
    const int64 kDefaultTilingFactor = 8;
    return options::LlvmIrGemvTilingFactor(hlo_module_config_)
        .value_or(kDefaultTilingFactor);
  }

  std::tuple<int64, int64, int64> GetGemmTileSize() const {
    // Tuned for broadwell - Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
    //
    // TODO(b/80093688): Tune for other architectures and centralize this
    // information in one place.
    const std::tuple<int64, int64, int64> kDefaultTileSize =
        std::tuple<int64, int64, int64>(11, 9, 1);
    return options::LlvmIrGemmTileSize(hlo_module_config_)
        .value_or(kDefaultTileSize);
  }

  const HloInstruction& dot_;
  const llvm_ir::IrArray& target_array_;
  const llvm_ir::IrArray& lhs_array_;
  const llvm_ir::IrArray& rhs_array_;
  const llvm_ir::IrArray* addend_array_;
  llvm::Value* executable_run_options_value_;
  llvm::IRBuilder<>* b_;
  const HloModuleConfig& hlo_module_config_;
  const TargetMachineFeatures& target_machine_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_H_
