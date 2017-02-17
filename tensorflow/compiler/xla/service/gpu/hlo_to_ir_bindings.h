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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_

#include <unordered_map>

#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace gpu {

// This class encapsulates the bindings between HloInstructions and LLVM IR
// values that represent their addresses.
class HloToIrBindings {
 public:
  HloToIrBindings(const HloModule& module,
                  const BufferAssignment* buffer_assignment,
                  llvm::IRBuilder<>* ir_builder, bool is_nested)
      : buffer_assignment_(buffer_assignment),
        is_nested_(is_nested),
        ir_builder_(ir_builder),
        alias_analysis_(module, *buffer_assignment_,
                        &ir_builder_->getContext()) {}

  void EmitBasePointersForHlos(
      tensorflow::gtl::ArraySlice<const HloInstruction*> io_hlos,
      tensorflow::gtl::ArraySlice<const HloInstruction*> non_io_hlos);

  // Rebinds the given HLO to the LLVM IR value that represent its address.
  void BindHloToIrValue(const HloInstruction& hlo, llvm::Value* ir_value);

  // Unbinds all IR values that's defined in an LLVM function, e.g., function
  // arguments and stack variables. Global variables will be kept in bindings_.
  //
  // This method is called after emitting code for each top-level HLO. The local
  // IR values are out of scope at that point and should not be used.
  void UnbindAllLocalIrValues();

  // Returns whether `hlo` is bound to an LLVM IR value.
  bool BoundToIrValue(const HloInstruction& hlo) const {
    return base_ptrs_.count(&hlo);
  }

  llvm::Value* GetTempBufferBase() const { return temp_buffer_base_; }

  // A helper method that returns the base pointer of the IrArray for "inst".
  llvm::Value* GetBasePointer(const HloInstruction& hlo) const {
    auto it = base_ptrs_.find(&hlo);
    CHECK(it != base_ptrs_.end());
    return it->second;
  }

  // Return the underlying IrArray of the output of the given instruction.
  llvm_ir::IrArray GetIrArray(const HloInstruction& hlo);

 private:
  // Emits IR to resolve (possibly) recursive GetTupleElement instructions.
  llvm::Value* EmitGetTupleElement(const HloInstruction* gte,
                                   llvm::Value* base_ptr);

  // Returns an llvm typed ir representation of 'ir_value' based on 'hlo' shape.
  llvm::Value* GetTypedIrValue(const HloInstruction& hlo,
                               llvm::Value* ir_value);

  const BufferAssignment* buffer_assignment_;

  const bool is_nested_;

  llvm::IRBuilder<>* ir_builder_;

  // Stores the underlying llvm::IrArray for each HloInstruction.
  std::unordered_map<const HloInstruction*, llvm::Value*> base_ptrs_;

  // The address of the memory block that contains all temporary buffers.
  llvm::Value* temp_buffer_base_;

  llvm_ir::AliasAnalysis alias_analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
