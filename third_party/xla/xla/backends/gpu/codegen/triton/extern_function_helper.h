/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_

#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

// Represents the different types of extern elementwise functions
// used in Triton XLA passes.

// Represents an atomic write operation
struct AtomicWriteInstruction {
  static constexpr llvm::StringRef kName = "atomicwrite";
  static constexpr int kNumArgs = 3;
  triton::MemSemantic semantic;
  triton::MemSyncScope scope;
  bool has_mask;

  bool operator==(const AtomicWriteInstruction& other) const {
    return semantic == other.semantic && scope == other.scope &&
           has_mask == other.has_mask;
  }
};

// Represents an atomic spin-wait operation
struct AtomicSpinWaitInstruction {
  static constexpr llvm::StringRef kName = "atomicspinwait";
  static constexpr int kNumArgs = 4;
  triton::MemSemantic semantic;
  triton::MemSyncScope scope;
  Comparator comparator;
  bool has_mask;

  bool operator==(const AtomicSpinWaitInstruction& other) const {
    return semantic == other.semantic && scope == other.scope &&
           comparator == other.comparator && has_mask == other.has_mask;
  }
};

// Represents a get thread ID operation
struct GetThreadIdInstruction {
  static constexpr llvm::StringRef kName = "getthreadid";
  static constexpr int kNumArgs = 0;
  bool operator==(const GetThreadIdInstruction&) const { return true; }
};

// Variant type that can hold any of the supported instruction types
using ExternFunctionInstruction =
    std::variant<AtomicWriteInstruction, AtomicSpinWaitInstruction,
                 GetThreadIdInstruction>;

// Parses a function name string into an ExternFunctionInstruction variant.
// Returns an error status if the function name is invalid or doesn't match
// any known pattern.
//
// Function name format: xla_<functionname>_<arg1>_<arg2>_...
// - Token[0]: "xla" (prefix)
// - Token[1]: function name
// - Token[2+]: arguments (if any)
//
// Supported patterns:
// - "xla_getthreadid" (2 tokens: no arguments)
// - "xla_atomicwrite_<semantic>_<scope>_<mask|nomask>" (5 tokens)
// - "xla_atomicspinwait_<semantic>_<scope>_<comparator>_<mask|nomask>" (6
//   tokens)
//
// Where:
// - <semantic>: relaxed, acquire, release, acqrel
// - <scope>: system, gpu, cta
// - <comparator>: eq, lt
// - <mask|nomask>: whether the operation has a mask as an operand
absl::StatusOr<ExternFunctionInstruction> ParseExternFunctionName(
    llvm::StringRef func_name);

// Serializes an ExternFunctionInstruction back to its string representation.
// This is the inverse of ParseExternFunctionName.
std::string SerializeExternFunctionName(
    const ExternFunctionInstruction& instruction);

// Validates that the memory semantic is appropriate for the instruction type.
// Returns an error status if validation fails.
absl::Status ValidateMemorySemantic(
    const ExternFunctionInstruction& instruction);

// Target backend for code generation
enum class TargetBackend {
  CUDA,
  ROCM,
};

// Parameters for creating LLVM operations from an instruction
struct LLVMOpCreationParams {
  mlir::OpBuilder& builder;
  mlir::Location loc;
  TargetBackend target;
  mlir::ValueRange
      operands;  // Operands from the call (ptr, value/expected, mask?)
};

// Creates the appropriate LLVM operations for the given instruction.
// This function generates the complete LLVM IR implementation for the
// instruction, including control flow for masked operations and loops for
// spin-wait operations.
//
// Returns the result value that should replace the original call operation.
// For operations that don't produce a meaningful result (like atomic_write),
// returns a poison value.
mlir::Value CreateLLVMOpsForInstruction(
    const ExternFunctionInstruction& instruction,
    const LLVMOpCreationParams& params);

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_
