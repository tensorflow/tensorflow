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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_

#include <unordered_map>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class ElementalIrEmitter {
 public:
  using HloToElementGeneratorMap =
      std::unordered_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  ElementalIrEmitter(const HloModuleConfig& hlo_module_config,
                     llvm::Module* module, llvm::IRBuilder<>* ir_builder)
      : ir_builder_(ir_builder),
        module_(module),
        hlo_module_config_(hlo_module_config) {}

  virtual ~ElementalIrEmitter() = default;

  virtual StatusOr<llvm::Value*> EmitUnaryOp(const HloInstruction* op,
                                             llvm::Value* operand_value) const;

  virtual StatusOr<llvm::Value*> EmitBinaryOp(const HloInstruction* op,
                                              llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) const;

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  virtual llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator) const;

  llvm::IRBuilder<>* ir_builder() const { return ir_builder_; }
  llvm::Module* module() const { return module_; }

 protected:
  virtual StatusOr<llvm::Value*> EmitIntegerUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value) const;

  virtual StatusOr<llvm::Value*> EmitFloatUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value) const;

  virtual StatusOr<llvm::Value*> EmitComplexUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value) const;

  virtual StatusOr<llvm::Value*> EmitIntegerBinaryOp(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value,
                                                     bool is_signed) const;

  virtual StatusOr<llvm::Value*> EmitFloatBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value,
      llvm::Value* rhs_value) const;

  virtual StatusOr<llvm::Value*> EmitComplexBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value,
      llvm::Value* rhs_value) const;

  virtual llvm::Value* EmitFloatMax(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value) const;

  virtual llvm::Value* EmitFloatMin(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value) const;

  virtual StatusOr<llvm::Value*> EmitErfInv(PrimitiveType prim_type,
                                            llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitErfcInv(PrimitiveType prim_type,
                                             llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                           llvm::Value* lhs,
                                           llvm::Value* rhs) const;

  virtual StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                         llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                         llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                         llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                         llvm::Value* value) const;

  virtual StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type,
                                         llvm::Value* lhs,
                                         llvm::Value* rhs) const;

  virtual StatusOr<llvm::Value*> EmitReducePrecision(const HloInstruction* hlo,
                                                     llvm::Value* x) const;

  virtual llvm::Value* EmitExtractReal(llvm::Value* value) const;
  virtual llvm::Value* EmitExtractImag(llvm::Value* value) const;

  // Composes a complex struct. imag may be nullptr for simple cast operations.
  llvm::Value* EmitComposeComplex(const HloInstruction* op, llvm::Value* real,
                                  llvm::Value* imag) const;

  // A helper method for MakeElementGenerator. Given an elementwise op `hlo` and
  // the target array index, computes the source array index of its
  // `operand_no`-th operand.
  //
  // Precondition: `hlo` is an elementwise op.
  llvm_ir::IrArray::Index ElementwiseSourceIndex(
      const llvm_ir::IrArray::Index& target_index, const HloInstruction& hlo,
      int64 operand_no) const;

  // Identifier of the thread unique among all threads on the device
  virtual llvm::Value* EmitThreadId() const {
    return ir_builder_->getIntN(128, 0);
  }

  llvm::IRBuilder<>* const ir_builder_;

  llvm::Module* module_;

  // The HloModuleConfig which gathers all settings and values which affect the
  // compiled executable outside of the HLO code itself.
  const HloModuleConfig& hlo_module_config_;

 private:
  // Returns a ElementGenerator for a RNG HloInstruction.
  llvm_ir::ElementGenerator MakeRngElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator) const;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
