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

#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

// Unlike IrEmitter, this creates host functions which emit IR to generate the
// output element at the given index. It is used to generate fused operations.
class FusedIrEmitter : public DfsHloVisitorWithDefault {
 public:
  using Generator = llvm_ir::ElementGenerator;

  FusedIrEmitter(tensorflow::gtl::ArraySlice<llvm_ir::IrArray> parameter_arrays,
                 ElementalIrEmitter* elemental_emitter)
      : parameter_arrays_(parameter_arrays),
        elemental_emitter_(elemental_emitter),
        ir_builder_(elemental_emitter->ir_builder()) {}

  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;

  Status HandleParameter(HloInstruction* parameter) override;

  Status FinishVisit(HloInstruction* root) override;

  // Returns the generator function for the root of the fused computation.
  Generator GetRootGenerator() const;

  // Returns the generator function for the given instruction.
  Generator GetGenerator(const HloInstruction* instruction) const;

 private:
  // Arrays of parameters of fusion instruction
  tensorflow::gtl::ArraySlice<llvm_ir::IrArray> parameter_arrays_;

  ElementalIrEmitter* elemental_emitter_;

  // This member will be set by FinishVisit and used in GetRootGenerator.
  const HloInstruction* fused_root_ = nullptr;

  // Borrowed
  llvm::IRBuilder<>* ir_builder_;

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
