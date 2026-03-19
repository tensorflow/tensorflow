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

#ifndef XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
#define XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

// This class encapsulates the bindings between HloInstructions and LLVM IR
// values that represent their addresses.
class HloToIrBindings {
 public:
  HloToIrBindings(llvm::IRBuilderBase* b, llvm::Module* llvm_module,
                  bool is_nested)
      : is_nested_(is_nested), b_(b), module_(llvm_module) {}

  void EmitBasePointersForHlos(
      absl::Span<const HloInstruction* const> io_hlos,
      absl::Span<const HloInstruction* const> non_io_hlos);

  // Rebinds the given HLO to the LLVM IR value that represent its address.
  void BindHloToIrValue(const HloInstruction& hlo, llvm::Value* ir_value,
                        ShapeIndexView shape_index = {});

  // Returns whether `hlo` is bound to an LLVM IR value.
  bool BoundToIrValue(const HloInstruction& hlo) const {
    return base_ptrs_.contains(&hlo);
  }

  // A helper method that returns the base pointer of the IrArray containing the
  // output of "inst".at the given ShapeIndex.
  llvm::Value* GetBasePointer(const HloInstruction& hlo,
                              ShapeIndexView shape_index = {}) const {
    auto it = base_ptrs_.find(&hlo);
    CHECK(it != base_ptrs_.end()) << hlo.ToString();
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
  const bool is_nested_;

  llvm::IRBuilderBase* b_;
  llvm::Module* module_;

  // Stores the underlying llvm::IrArray for each HloInstruction.
  // For an instruction that generates multiple outputs, the root will be a
  // tuple shape. The IrArray for each element output is stored in the subnode
  // in the ShapeTree.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<llvm::Value*>>
      base_ptrs_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_TO_IR_BINDINGS_H_
