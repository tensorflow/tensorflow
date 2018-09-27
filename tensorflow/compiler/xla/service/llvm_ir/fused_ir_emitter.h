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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_

#include <map>
#include <unordered_map>

#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_tiling.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// FusedIrEmitter is used to generate code for fusion nodes.
//
// Unlike IrEmitter and its ilk, which directly create LLVM IR in an LLVM
// Module, FusedIrEmitter is better understood as "IR generator generator".
// FusedIrEmitter recursively creates a generator (a host function) which the
// compiler can invoke at a later time.  Invoking the generator emits LLVM IR
// that, when run, produces the value at a particular index of the output.
//
// After building this generator, the compiler creates a loop (or its moral
// equivalent, e.g. a GPU kernel) and calls the generator from within the loop.
// This generates code that produces each element of the output.
//
// This class handles both vanilla fusion and multi-output fusion.  In the MOF
// case, the fusion node ends with a kTuple instruction, and the generator
// created produces an LLVM struct with N elements, one for each element of the
// arrays in the tuple.  It follows that the arrays in the tuple must have the
// same length.
class FusedIrEmitter : public DfsHloVisitorWithDefault {
 public:
  using Generator = llvm_ir::ElementGenerator;

  FusedIrEmitter(absl::Span<const llvm_ir::IrArray> parameter_arrays,
                 ElementalIrEmitter* elemental_emitter)
      : parameter_arrays_(parameter_arrays),
        tiled_parameter_info_(nullptr),
        elemental_emitter_(elemental_emitter),
        b_(elemental_emitter->b()),
        module_(elemental_emitter->module()) {}

  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleParameter(HloInstruction* parameter) override;

  // Emits the ir value for each element in the tuple.
  Status HandleTuple(HloInstruction* tuple) override;

  Status FinishVisit(HloInstruction* root) override;

  // Returns the generator function for the root of the fused computation.
  Generator GetRootGenerator() const;

  // Returns the generator function for the given instruction.
  Generator GetGenerator(const HloInstruction* instruction) const;

  // Returns the ir value for instruction 'hlo'.
  llvm::Value* GetIrValueForGTE(const HloInstruction* hlo) const {
    auto it = gte_values_.find(hlo);
    CHECK(it != gte_values_.end());
    return it->second;
  }

  void SetTiledParameterInfo(const llvm_ir::TiledParameterInfo* info) {
    tiled_parameter_info_ = info;
  }

 private:
  // Arrays of parameters of fusion instruction
  absl::Span<const llvm_ir::IrArray> parameter_arrays_;
  const llvm_ir::TiledParameterInfo* tiled_parameter_info_;

  ElementalIrEmitter* elemental_emitter_;

  // This member will be set by FinishVisit and used in GetRootGenerator.
  const HloInstruction* fused_root_ = nullptr;

  // Borrowed
  llvm::IRBuilder<>* b_;
  llvm::Module* module_;

  // Map from instruction pointers to functions to generate elements of their
  // outputs
  std::unordered_map<const HloInstruction*, Generator> generators_;

  // Cache of generated values, lest we regenerate an element of a node with
  // multiple outgoing edges
  std::unordered_map<const HloInstruction*,
                     std::map<std::vector<llvm::Value*>, llvm::Value*>>
      generated_value_cache_;

  // Stores ir values required to emit fused (and possibly nested)
  // GetTupleElement instructions.
  std::unordered_map<const HloInstruction*, llvm::Value*> gte_values_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
