/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_DOT_OP_EMITTER_H_
#define XLA_SERVICE_CPU_DOT_OP_EMITTER_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

// Dictates how a dot operation is implemented.
enum class DotImplementationStrategy {
  // The dot operation is lowered into LLVM IR that implements a naive nested
  // loop that computes the result one element at a time.  This is our
  // "fallback"; we don't really want this to kick in for any non-trivial dot
  // operation.
  kNaiveLlvmIr,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Vector operation.  This strategy also allows fusing in a bias add
  // into the dot.  The matrix can be row major or column major, both are
  // supported.
  kTiledLlvmIrGemv,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Matrix operation.  No fusions are supported.  The two inputs
  // and the output have to be row major.
  kTiledLlvmIrGemm,

  // The dot operation is lowered into a call into an Eigen routine.  No fusions
  // are supported today.  The two inputs and the output have to be row major.
  // However, we do allow transposing either the LHS or the RHS as part of the
  // GEMM -- we expose this flexibility as flexibility in the contraction
  // dimensions, but we can also see this as flexibility in the input layouts.
  kEigen,
};

// Represents a dot operation.  We use this in lieu of an `HloInstruction`
// because we want to be able to create this for the "inner" dot operation in a
// batch dot, for which there is no separate HLO instruction.
struct DotInfo {
  Shape lhs_shape;
  Shape rhs_shape;
  Shape result_shape;
  DotDimensionNumbers dim_nums;

  DotInfo() = default;

  explicit DotInfo(const HloInstruction& instr) {
    CHECK_EQ(instr.opcode(), HloOpcode::kDot);
    lhs_shape = instr.operand(0)->shape();
    rhs_shape = instr.operand(1)->shape();
    result_shape = instr.shape();
    dim_nums = instr.dot_dimension_numbers();
  }
};

// Returns true if `instr` is a batch dot.
bool IsBatchDot(const HloInstruction& instr);

// Returns true if `dot_info` is a batch dot.
bool IsBatchDot(const DotInfo& dot_info);

// Returns `DotInfo` for the inner dot operation of the `batch_dot`.
DotInfo InnerDotInfo(const DotInfo& batch_dot);

// Returns the implementation strategy for a dot with the configuration
// `dot_info`.
DotImplementationStrategy GetDotImplementationStrategy(
    const HloModuleConfig& config, const HloInstruction& instr,
    const TargetMachineFeatures& target_machine_features,
    bool allow_runtime_calls);

// Returns true if the two operands and the output of `dot_instr` must have row
// major layout.
bool DotOperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features,
    bool allow_runtime_calls);

// Returns true our lowering strategy for `dot_instr` can fold in transposes to
// the either of the inputs.
bool DotImplementationCanHandleTranspose(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features,
    bool allow_runtime_calls);

// Returns the index for an operand to `hlo` that should ideally be column
// major.  Returns nullopt if there is no such operand or if `hlo` is not a dot
// or a fusion containing a dot.
std::optional<int64_t> ProfitableToMakeDotOperandColumnMajor(
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
//
// If `allow_runtime_calls` is false and DotEmitter tries to emit a call to a
// runtime API, it will return an error.
absl::Status EmitDotOperation(
    const HloInstruction& dot, const llvm_ir::IrArray& target_array,
    const llvm_ir::IrArray& lhs_array, const llvm_ir::IrArray& rhs_array,
    const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilderBase* b,
    const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features,
    bool allow_runtime_calls = true);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_DOT_OP_EMITTER_H_
