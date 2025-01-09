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

#ifndef XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_

#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/loop_emitter.h"

namespace xla {

class ElementalIrEmitter : public IrBuilderMixin<ElementalIrEmitter> {
 public:
  struct Options {
    // Instead of relying on builtin `fpext` and `fpcast` emit a bitcast and
    // truncate to convert f32 to bf16 (and emit extend to convert bf16 to f32).
    bool xla_cpu_use_truncate_f32_to_bf16_conversion = false;
  };

  using HloToElementGeneratorMap =
      absl::flat_hash_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilderBase* b,
                     const Options& options)
      : b_(b), module_(module), options_(options) {}

  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilderBase* b)
      : ElementalIrEmitter(module, b, Options()) {}

  virtual ~ElementalIrEmitter() = default;

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator);

  llvm::IRBuilderBase* b() { return b_; }

  // builder() is for IrBuilderMixin.
  llvm::IRBuilderBase* builder() { return b_; }

  llvm::Module* module() { return module_; }

  // Returns which ops invalidate the cache of emitted instructions by creating
  // a new BasicBlock and setting the insertion point to the newly created
  // BasicBlock. We can only reuse cached values if they were emitted in the
  // same BasicBlock as the current BasicBlock.
  static bool OpInvalidatesCache(const HloInstruction* hlo);

 protected:
  virtual llvm_ir::IrArray::Index GetSourceIndexOfBitcast(
      const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
    return index.SourceIndexOfBitcast(hlo->shape(), hlo->operand(0)->shape(),
                                      b_);
  }

  virtual absl::StatusOr<llvm::Value*> EmitFloatBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual llvm::Value* EmitExtractReal(llvm::Value* value);
  virtual llvm::Value* EmitExtractImag(llvm::Value* value);

 private:
  virtual absl::StatusOr<llvm::Value*> EmitUnaryOp(const HloInstruction* op,
                                                   llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitBinaryOp(const HloInstruction* op,
                                                    llvm::Value* lhs_value,
                                                    llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitIntegerUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitFloatUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value);

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
  llvm::Value* EmitIntegerPow(llvm::Value* lhs, llvm::Value* rhs,
                              bool is_signed);

  virtual absl::StatusOr<llvm::Value*> EmitPredBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitIntegerBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value,
      bool is_signed);

  virtual absl::StatusOr<llvm::Value*> EmitComplexBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual llvm::Value* EmitFloatMax(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value,
                                    absl::string_view name);

  virtual llvm::Value* EmitFloatMin(llvm::Value* lhs_value,
                                    llvm::Value* rhs_value,
                                    absl::string_view name);

  llvm::Value* EmitIntegralMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  llvm::Value* EmitIntegralMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                               bool is_signed);

  virtual absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                                 llvm::Value* lhs,
                                                 llvm::Value* rhs,
                                                 absl::string_view name);

  virtual absl::StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                               llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitSqrt(PrimitiveType prim_type,
                                                llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitCbrt(PrimitiveType prim_type,
                                                llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitRsqrt(PrimitiveType prim_type,
                                                 llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitLog1p(PrimitiveType prim_type,
                                                 llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                               llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                               llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitCosm1(PrimitiveType prim_type,
                                                 llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitTan(PrimitiveType prim_type,
                                               llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                               llvm::Value* value,
                                               absl::string_view name);

  virtual absl::StatusOr<llvm::Value*> EmitExpm1(PrimitiveType prim_type,
                                                 llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type,
                                               llvm::Value* lhs,
                                               llvm::Value* rhs,
                                               absl::string_view name);

  virtual absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                               llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                                llvm::Value* value);

  virtual absl::StatusOr<llvm::Value*> EmitReducePrecision(
      const HloInstruction* hlo, llvm::Value* x);

  virtual absl::StatusOr<std::tuple<llvm::Value*, llvm::Value*, llvm::Value*>>
  EmitComplexAbsHelper(PrimitiveType prim_type, llvm::Value* real,
                       llvm::Value* imag, bool return_sqrt);

  virtual absl::StatusOr<llvm::Value*> EmitComplexAbs(
      PrimitiveType prim_type, llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitSqrtComplexAbs(
      PrimitiveType prim_type, llvm::Value* operand_value);
  virtual absl::StatusOr<llvm::Value*> EmitRsqrtComplexAbs(
      PrimitiveType prim_type, llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexAdd(const HloInstruction* op,
                                                      llvm::Value* lhs_value,
                                                      llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexSubtract(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexMultiply(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexDivide(
      const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexLog(
      const HloInstruction* op, llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexSqrt(
      const HloInstruction* op, PrimitiveType prim_type,
      llvm::Value* operand_value);

  virtual absl::StatusOr<llvm::Value*> EmitComplexRsqrt(
      const HloInstruction* op, PrimitiveType prim_type,
      llvm::Value* operand_value);

  absl::StatusOr<llvm::Value*> EmitAccumResult(
      absl::Span<llvm::Value* const> accumulator_addrs,
      llvm::ArrayRef<llvm::Type*> accumulator_types, bool is_variadic);

  // Composes a complex struct. imag may be nullptr for simple cast operations.
  llvm::Value* EmitComposeComplex(const HloInstruction* op, llvm::Value* real,
                                  llvm::Value* imag);

  // Emit `accumulator + lhs * rhs` for the given primitive type.
  llvm::Value* EmitMulAdd(llvm::Value* lhs, llvm::Value* rhs,
                          llvm::Value* accumulator,
                          xla::PrimitiveType primitive_type);

  absl::StatusOr<llvm::Value*> EmitElementalSelect(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalClamp(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalConcatenate(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& target_index);

  absl::StatusOr<llvm::Value*> EmitElementalDynamicSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalGather(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalDynamicUpdateSlice(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalPad(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& padded_index);

  absl::StatusOr<llvm::Value*> EmitElementalDot(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& dot_result_index);

  virtual absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) = 0;

  absl::StatusOr<llvm::Value*> EmitElementalMap(
      const HloMapInstruction* map_instr,
      absl::Span<llvm::Value* const> elemental_operands);

  absl::StatusOr<llvm::Value*> EmitElementalReduceWindow(
      const HloReduceWindowInstruction* reduce_window,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  absl::StatusOr<llvm::Value*> EmitElementalReduce(
      const HloReduceInstruction* reduce,
      std::vector<llvm_ir::ElementGenerator> input_generators,
      std::vector<llvm_ir::ElementGenerator> initial_value_generators,
      const llvm_ir::IrArray::Index& index);

  virtual absl::StatusOr<llvm::Value*> EmitConvolution(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator,
      const llvm_ir::IrArray::Index& index);

  // Computes the complex power function.
  absl::StatusOr<llvm::Value*> EmitComplexPower(const HloInstruction* op,
                                                llvm::Value* lhs_value,
                                                llvm::Value* rhs_value);

  // Evaluates a polynomial using Horner's method.
  absl::StatusOr<llvm::Value*> EvaluatePolynomial(
      llvm::Type* type, llvm::Value* x, absl::Span<const double> coefficients);

  virtual bool fast_min_max() = 0;

  llvm::IRBuilderBase* const b_;

  llvm::Module* module_;

  Options options_;

  friend class ElementalIrEmitterForTests;
};

// Allow to instantiate IR emitter in tests.
class ElementalIrEmitterForTests : public ElementalIrEmitter {
 public:
  ElementalIrEmitterForTests(llvm::Module* module, llvm::IRBuilderBase* builder)
      : ElementalIrEmitter(module, builder) {}

  absl::Status TestElementalDot(const HloInstruction* hlo,
                                const llvm_ir::IrArray::Index& index) {
    return EmitElementalDot(hlo, generator_map_, index).status();
  }

 private:
  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    return absl::UnimplementedError("");
  }
  bool fast_min_max() override { return false; }

  HloToElementGeneratorMap generator_map_;
};

absl::StatusOr<llvm::Value*> EmitReducePrecisionIR(
    PrimitiveType src_ty, llvm::Value* x, int64_t dest_exponent_bits,
    int64_t dest_mantissa_bits, bool quiet_nans, llvm::IRBuilderBase* b);

}  // namespace xla

#endif  // XLA_SERVICE_ELEMENTAL_IR_EMITTER_H_
