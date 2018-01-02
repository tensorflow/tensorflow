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

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
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
                  llvm::IRBuilder<>* ir_builder, llvm::Module* llvm_module,
                  bool is_nested)
      : buffer_assignment_(buffer_assignment),
        is_nested_(is_nested),
        ir_builder_(ir_builder),
        module_(llvm_module),
        alias_analysis_(module, *buffer_assignment_,
                        &ir_builder_->getContext()) {}

  void EmitBasePointersForHlos(
      tensorflow::gtl::ArraySlice<const HloInstruction*> io_hlos,
      tensorflow::gtl::ArraySlice<const HloInstruction*> non_io_hlos);

  // Rebinds the given HLO to the LLVM IR value that represent its address.
  void BindHloToIrValue(const HloInstruction& hlo, llvm::Value* ir_value,
                        const ShapeIndex& shape_index = {});

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

  // A helper method that returns the base pointer of the IrArray containing the
  // output of "inst".at the given ShapeIndex.
  llvm::Value* GetBasePointer(const HloInstruction& hlo,
                              const ShapeIndex& shape_index = {}) const {
    auto it = base_ptrs_.find(&hlo);
    CHECK(it != base_ptrs_.end());
    return it->second.element(shape_index);
  }

  // Returns the IrArray which contains the output of hlo.
  //
  // consumer is the HLO in which this IrArray is used -- we use this to (try
  // to) add metadata indicating that the array is invariant within consumer.
  //
  // To get the buffer into which hlo should write its own output, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& hlo,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {});

 private:
  // Emits IR to resolve (possibly) recursive GetTupleElement instructions.
  llvm::Value* EmitGetTupleElement(const HloInstruction* gte,
                                   llvm::Value* base_ptr);

  // Returns an llvm typed ir representation of 'ir_value' based on 'hlo' shape.
  llvm::Value* GetTypedIrValue(const HloInstruction& hlo,
                               const ShapeIndex& shape_index,
                               llvm::Value* ir_value);

  const BufferAssignment* buffer_assignment_;

  const bool is_nested_;

  llvm::IRBuilder<>* ir_builder_;
  llvm::Module* module_;

  // Stores the underlying llvm::IrArray for each HloInstruction.
  // For an instruction that generates multiple outputs, the root will be a
  // tuple shape. The IrArray for each element output is stored in the subnode
  // in the ShapeTree.
  std::unordered_map<const HloInstruction*, ShapeTree<llvm::Value*>> base_ptrs_;

  // The address of the memory block that contains all temporary buffers.
  llvm::Value* temp_buffer_base_;

  llvm_ir::AliasAnalysis alias_analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
