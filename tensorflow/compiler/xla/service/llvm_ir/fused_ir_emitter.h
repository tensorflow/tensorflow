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

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
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
class FusedIrEmitter : public ConstDfsHloVisitorWithDefault {
 public:
  using IndexedGenerator = llvm_ir::ElementGenerator;
  using NonIndexedGenerator = std::function<StatusOr<llvm::Value*>()>;
  using GeneratorForOperandIrArrays =
      std::function<std::vector<llvm_ir::IrArray>()>;

  FusedIrEmitter(GeneratorForOperandIrArrays operand_arrays_generator,
                 ElementalIrEmitter* elemental_emitter,
                 llvm::Value* thread_id_x = nullptr,
                 llvm::Value* thread_id_y = nullptr,
                 absl::Span<llvm::Value* const> param_shmem_buffers = {})
      : operand_arrays_(),
        operand_arrays_generator_(std::move(operand_arrays_generator)),
        thread_id_x_(thread_id_x),
        thread_id_y_(thread_id_y),
        param_shmem_buffers_(param_shmem_buffers.begin(),
                             param_shmem_buffers.end()),
        elemental_emitter_(elemental_emitter),
        b_(elemental_emitter->b()),
        module_(elemental_emitter->module()) {}

  Status DefaultAction(const HloInstruction* hlo) override;

  Status HandleConstant(const HloInstruction* constant) override;

  Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;

  Status HandleParameter(const HloInstruction* parameter) override;

  // Emits the ir value for each element in the tuple.
  Status HandleTuple(const HloInstruction* tuple) override;

  Status FinishVisit(const HloInstruction* root) override;

  // Returns the generator function for the root of the fused computation.
  IndexedGenerator GetRootGenerator() const;

  // Returns the generator function for the given instruction.
  IndexedGenerator GetGenerator(const HloInstruction* instruction) const;

  // Evaluates whether fusing 'producer' into 'consumer' might cause exponential
  // behavior in FusedIrEmitter. We currently can have exponential time/memory
  // requirements for emitting certain fusion kernels, in which case we don't
  // want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  static bool IsFusedIrEmitterInefficient(const HloInstruction* consumer,
                                          const HloInstruction* producer);

 protected:
  // Returns the IrArrays for the fusion instruction operands.
  llvm_ir::IrArray& GetIrArrayForFusedParameter(int64 parameter_number) {
    if (!operand_arrays_.has_value()) {
      operand_arrays_ = operand_arrays_generator_();
    }
    return operand_arrays_.value()[parameter_number];
  }

  llvm::Value* GetBasePointerForFusedParameter(int64 parameter_number) {
    return GetIrArrayForFusedParameter(parameter_number).GetBasePointer();
  }

 private:
  // IrArrays for the fusion instruction operands, whose base addresses are the
  // base address of the corresponding parameters in the fused computation.
  absl::optional<std::vector<llvm_ir::IrArray>> operand_arrays_;
  GeneratorForOperandIrArrays operand_arrays_generator_;

  // The x coordinate within a tile.
  llvm::Value* thread_id_x_;

  // The y coordinate within a tile.
  llvm::Value* thread_id_y_;

  // Param_buffers_[i] stores the tile buffer for the ith parameter or nullptr
  // if the parameter is not tiled.
  std::vector<llvm::Value*> param_shmem_buffers_;

  ElementalIrEmitter* elemental_emitter_;

  // This member will be set by FinishVisit and used in GetRootGenerator.
  const HloInstruction* fused_root_ = nullptr;

  // Borrowed
  llvm::IRBuilder<>* b_;
  llvm::Module* module_;

  // Map from instructions to functions that generate code for the output
  // elements. If an instruction is a GetTupleElement instruction, the
  // instruction produces non-tuple result.
  std::unordered_map<const HloInstruction*, IndexedGenerator>
      indexed_generators_;

  // Map from tuple-result-producing GetTupleELement instructions to functions
  // that generate the base pointers for the output elements. This is used to
  // support the translation of nested GetTupleElement instructions.
  std::unordered_map<const HloInstruction*, NonIndexedGenerator>
      non_indexed_generators_;

  // Cache of generated values, lest we regenerate an element of a node with
  // multiple outgoing edges
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<std::vector<llvm::Value*>, llvm::Value*>>
      generated_value_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_FUSED_IR_EMITTER_H_
