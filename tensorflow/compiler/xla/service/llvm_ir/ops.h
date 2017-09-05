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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

// Selection among tuples is special in how it's lowered, because a tuple is not
// an HLO array.
//
//      tuple_on_true                     tuple_on_false
//           |                                 |
//           V                                 V
// ------------------------          ------------------------
// | address of element 0 |          | address of element 0 |
// |----------------------|          |----------------------|
// | address of element 1 |          | address of element 1 |
// |----------------------|          |----------------------|
// | address of element 2 |          | address of element 2 |
// ------------------------          ------------------------
//                       \            /
//                        \          /
//                         ----------
//         pred ---------> | select |
//                         ----------
//                             |
//                             V
//      output ----> ------------------------
//                   | address of element 0 |
//                   |----------------------|
//                   | address of element 1 |
//                   |----------------------|
//                   | address of element 2 |
//                   ------------------------
//
// Only the addresses are copied to the output. For each element, we emit a copy
// of the address from the corresponding element in either
// tuple_on_true or tuple_on_false:
//   output[i] = pred ? tuple_on_true[i] : tuple_on_false[i]
void EmitTupleSelect(IrArray select, IrArray pred, llvm::Value* on_true,
                     llvm::Value* on_false, llvm::IRBuilder<>* ir_builder);

// A tuple is an array of pointers, one for each operand. Each pointer points to
// the output buffer of its corresponding operand.
void EmitTuple(IrArray tuple,
               tensorflow::gtl::ArraySlice<llvm::Value*> operands,
               llvm::IRBuilder<>* ir_builder);

// A tuple is an array of pointers, one for each operand. Each pointer points to
// the output buffer of its corresponding operand. A GetTupleElement instruction
// forwards the pointer to underlying tuple element buffer at the given index.
// Returns an llvm value representing a pointer to the tuple element buffer.
llvm::Value* EmitGetTupleElement(const Shape& target_shape, int64 index,
                                 int alignment, llvm::Value* operand,
                                 llvm::IRBuilder<>* ir_builder);
}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_OPS_H_
