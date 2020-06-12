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
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class ElementalIrEmitter : public IrBuilderMixin<ElementalIrEmitter> {
 public:
  using HloToElementGeneratorMap =
      std::unordered_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  ElementalIrEmitter(const HloModuleConfig& hlo_module_config,
                     llvm::Module* module, llvm::IRBuilder<>* b)
      : b_(b), module_(module), hlo_module_config_(hlo_module_config) {}

  virtual ~ElementalIrEmitter() = default;

  virtual StatusOr<llvm::Value*> EmitUnaryOp(const HloInstruction* op,
                                             llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitBinaryOp(const HloInstruction* op,
                                              llvm::Value* lhs_value,
                                              llvm::Value* rhs_value);

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  virtual llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator);

  llvm::IRBuilder<>* b() { return b_; }

  // builder() is for IrBuilderMixin.
  llvm::IRBuilder<>* builder() { return b_; }

  llvm::Module* module() { return module_; }

 protected:
  virtual StatusOr<llvm::Value*> EmitIntegerUnaryOp(const HloInstruction* op,
                                                    llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitFloatUnaryOp(const HloInstruction* op,
                                                  llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexUnaryOp(const HloInstruction* op,
                                                    llvm::Value* operand_value);

  llvm::Value* IsZero(llvm::Value* v);
  llvm::Value* IsIntMinDivisionOverflow(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* GetZero(llvm::Type* type);
  llvm::Value* GetOne(llvm::Type* type);
  llvm::Value* GetIntSMin(llvm::Type* type);
  llvm::Value* GetMinusOne(llvm::Type* type);

  llvm::Value* EmitIntegerDivide(llvm::Value* lhs, llvm::Value* rhs,
                                 bool is_signed);
  llvm::Value* EmitIntegerRemainder(llvm::Value* lhs, llvm::Value* rhs,
                                    bool is_signed);

  virtual StatusOr<llvm::Value*> EmitIntegerBinaryOp(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value,
                                                     bool is_signed);

  virtual StatusOr<llvm::Value*> EmitFloatBinaryOp(const HloInstruction* op,
                                                   llvm::Value* lhs_value,
                                                   llvm::Value* rhs_value);

  virtual StatusOr<llvm::Value*> EmitComplexBinaryOp(const HloInstruction* op,
                                                     llvm::Value* lhs_value,
                                                     llvm::Value* rhs_value);

  virtual llvm::Value* EmitFloatMax(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value);

  virtual llvm::Value* EmitFloatMin(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value);

  llvm::Value* EmitIntegralMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  llvm::Value* EmitIntegralMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  virtual StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                           llvm::Value* lhs, llvm::Value* rhs);

  virtual StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitSqrt(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitCbrt(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitRsqrt(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitLog1p(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                         llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitExpm1(PrimitiveType prim_type,
                                           llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs);

  virtual StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                          llvm::Value* value);

  virtual StatusOr<llvm::Value*> EmitReducePrecision(const HloInstruction* hlo,
                                                     llvm::Value* x);

  virtual StatusOr<std::tuple<llvm::Value*, llvm::Value*, llvm::Value*>>
  EmitComplexAbsHelper(PrimitiveType prim_type, llvm::Value* operand_value,
                       bool return_sqrt);

  virtual StatusOr<llvm::Value*> EmitComplexAbs(PrimitiveType prim_type,
                                                llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitSqrtComplexAbs(PrimitiveType prim_type,
                                                    llvm::Value* operand_value);
  virtual StatusOr<llvm::Value*> EmitRsqrtComplexAbs(
      PrimitiveType prim_type, llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexSqrt(const HloInstruction* op,
                                                 PrimitiveType prim_type,
                                                 llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexCbrt(const HloInstruction* op,
                                                 PrimitiveType prim_type,
                                                 llvm::Value* operand_value);

  virtual StatusOr<llvm::Value*> EmitComplexRsqrt(const HloInstruction* op,
                                                  PrimitiveType prim_type,
                                                  llvm::Value* operand_value);

  virtual llvm::Value* EmitExtractReal(llvm::Value* value);
  virtual llvm::Value* EmitExtractImag(llvm::Value* value);

  // Composes a complex struct. imag may be nullptr for simple cast operations.
  llvm::Value* EmitComposeComplex(const HloInstruction* op, llvm::Value* real,
                                  llvm::Value* imag);

  // Identifier of the thread unique among all threads on the device
  virtual llvm::Value* EmitThreadId() { return b_->getIntN(128, 0); }

  StatusOr<llvm::Value*> EmitElementalSelect(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalClamp(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalConcatenate(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& target_index);

  StatusOr<llvm::Value*> EmitElementalDynamicSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalGather(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalDynamicUpdateSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalPad(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& padded_index);

  StatusOr<llvm::Value*> EmitElementalDot(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& dot_result_index);

  virtual StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name) = 0;

  StatusOr<llvm::Value*> EmitElementalMap(
      const HloMapInstruction* map_instr,
      absl::Span<llvm::Value* const> elemental_operands);

  StatusOr<llvm::Value*> EmitElementalReduceWindow(
      const HloReduceWindowInstruction* reduce_window,
      const llvm_ir::ElementGenerator& input_generator,
      const llvm_ir::ElementGenerator& initial_value_generator,
      const llvm_ir::IrArray::Index& index);

  StatusOr<llvm::Value*> EmitElementalReduce(
      const HloReduceInstruction* reduce,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  llvm::IRBuilder<>* const b_;

  llvm::Module* module_;

  // The HloModuleConfig which gathers all settings and values which affect the
  // compiled executable outside of the HLO code itself.
  const HloModuleConfig& hlo_module_config_;

 private:
  // Computes the complex power function, returns (a + i*b)^(c + i*d).
  StatusOr<llvm::Value*> EmitComplexPower(const HloInstruction* op,
                                          llvm::Value* a, llvm::Value* b,
                                          llvm::Value* c, llvm::Value* d);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
