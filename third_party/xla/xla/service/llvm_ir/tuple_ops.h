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

#ifndef XLA_SERVICE_LLVM_IR_TUPLE_OPS_H_
#define XLA_SERVICE_LLVM_IR_TUPLE_OPS_H_

#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"

// Utilities for emitting LLVM IR related to HLO tuples.

namespace xla {
namespace llvm_ir {

// A tuple is an array of pointers, one for each operand. Each pointer points to
// the output buffer of its corresponding operand.
void EmitTuple(const IrArray& tuple, absl::Span<llvm::Value* const> operands,
               llvm::IRBuilder<>* b);

// Emits one alloca for each element in the tuple of shape tuple_shape,
// returns the emitted allocas.
// Precondition: tuple_shape should be a tuple of scalars.
std::vector<llvm::Value*> EmitTupleAllocasAtFunctionEntry(
    const Shape& tuple_shape, llvm::IRBuilder<>* b);

// Similar to EmitTuple above, except that the output buffers are provided in
// the form of IrArray.
void EmitTuple(const IrArray& tuple, absl::Span<const IrArray> buffers,
               llvm::IRBuilder<>* b);

// A tuple is an array of pointers, one for each operand. Each pointer points to
// the output buffer of its corresponding operand. A GetTupleElement instruction
// forwards the pointer to underlying tuple element buffer at the given index.
// Returns an llvm value representing a pointer to the tuple element buffer.
llvm::Value* EmitGetTupleElement(const Shape& target_shape, int64_t index,
                                 int alignment, llvm::Value* operand,
                                 llvm::Type* operand_pointee_type,
                                 llvm::IRBuilder<>* b);
}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_TUPLE_OPS_H_
