//===- BuiltinOps.h - Builtin MLIR Operations -----------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines convenience types for working with builtin operations
// in the MLIR instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINOPS_H
#define MLIR_IR_BUILTINOPS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class Builder;

class BuiltinDialect : public Dialect {
public:
  BuiltinDialect(MLIRContext *context);
};

/// The "affine_apply" operation applies an affine map to a list of operands,
/// yielding a list of results. The operand and result list sizes must be the
/// same. All operands and results are of type 'Index'. This operation
/// requires a single affine map attribute named "map".
/// For example:
///
///   %y = "affine_apply" (%x) { map: (d0) -> (d0 + 1) } :
///          (index) -> (index)
///
/// equivalently:
///
///   #map42 = (d0)->(d0+1)
///   %y = affine_apply #map42(%x)
///
class AffineApplyOp
    : public Op<AffineApplyOp, OpTrait::VariadicOperands,
                OpTrait::VariadicResults, OpTrait::HasNoSideEffect> {
public:
  /// Builds an affine apply op with the specified map and operands.
  static void build(Builder *builder, OperationState *result, AffineMap map,
                    ArrayRef<Value *> operands);

  /// Returns the affine map to be applied by this operation.
  AffineMap getAffineMap() const {
    return getAttrOfType<AffineMapAttr>("map").getValue();
  }

  /// Returns true if the result of this operation can be used as dimension id.
  bool isValidDim() const;

  /// Returns true if the result of this operation is a symbol.
  bool isValidSymbol() const;

  static StringRef getOperationName() { return "affine_apply"; }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;
  bool constantFold(ArrayRef<Attribute> operandConstants,
                    SmallVectorImpl<Attribute> &results,
                    MLIRContext *context) const;

  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

private:
  friend class OperationInst;
  explicit AffineApplyOp(const OperationInst *state) : Op(state) {}
};

/// The "br" operation represents a branch instruction in a CFG function.
/// The operation takes variable number of operands and produces no results.
/// The operand number and types for each successor must match the
/// arguments of the block successor. For example:
///
///   bb2:
///      %2 = call @someFn()
///      br bb3(%2 : tensor<*xf32>)
///   bb3(%3: tensor<*xf32>):
///
class BranchOp : public Op<BranchOp, OpTrait::VariadicOperands,
                           OpTrait::ZeroResult, OpTrait::IsTerminator> {
public:
  static StringRef getOperationName() { return "br"; }

  static void build(Builder *builder, OperationState *result, Block *dest,
                    ArrayRef<Value *> operands = {});

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

  /// Return the block this branch jumps to.
  Block *getDest();
  const Block *getDest() const {
    return const_cast<BranchOp *>(this)->getDest();
  }
  void setDest(Block *block);

  /// Erase the operand at 'index' from the operand list.
  void eraseOperand(unsigned index);

private:
  friend class OperationInst;
  explicit BranchOp(const OperationInst *state) : Op(state) {}
};

/// The "cond_br" operation represents a conditional branch instruction in a
/// CFG function. The operation takes variable number of operands and produces
/// no results. The operand number and types for each successor must match the
//  arguments of the block successor. For example:
///
///   bb0:
///      %0 = extract_element %arg0[] : tensor<i1>
///      cond_br %0, bb1, bb2
///   bb1:
///      ...
///   bb2:
///      ...
///
class CondBranchOp : public Op<CondBranchOp, OpTrait::AtLeastNOperands<1>::Impl,
                               OpTrait::ZeroResult, OpTrait::IsTerminator> {
  // These are the indices into the dests list.
  enum { trueIndex = 0, falseIndex = 1 };

  /// The operands list of a conditional branch operation is layed out as
  /// follows:
  /// { condition, [true_operands], [false_operands] }
public:
  static StringRef getOperationName() { return "cond_br"; }

  static void build(Builder *builder, OperationState *result, Value *condition,
                    Block *trueDest, ArrayRef<Value *> trueOperands,
                    Block *falseDest, ArrayRef<Value *> falseOperands);

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

  // The condition operand is the first operand in the list.
  Value *getCondition() { return getOperand(0); }
  const Value *getCondition() const { return getOperand(0); }

  /// Return the destination if the condition is true.
  Block *getTrueDest();
  const Block *getTrueDest() const {
    return const_cast<CondBranchOp *>(this)->getTrueDest();
  }

  /// Return the destination if the condition is false.
  Block *getFalseDest();
  const Block *getFalseDest() const {
    return const_cast<CondBranchOp *>(this)->getFalseDest();
  }

  // Accessors for operands to the 'true' destination.
  Value *getTrueOperand(unsigned idx) {
    assert(idx < getNumTrueOperands());
    return getOperand(getTrueDestOperandIndex() + idx);
  }
  const Value *getTrueOperand(unsigned idx) const {
    return const_cast<CondBranchOp *>(this)->getTrueOperand(idx);
  }
  void setTrueOperand(unsigned idx, Value *value) {
    assert(idx < getNumTrueOperands());
    setOperand(getTrueDestOperandIndex() + idx, value);
  }

  operand_iterator true_operand_begin() {
    return operand_begin() + getTrueDestOperandIndex();
  }
  operand_iterator true_operand_end() {
    return true_operand_begin() + getNumTrueOperands();
  }
  llvm::iterator_range<operand_iterator> getTrueOperands() {
    return {true_operand_begin(), true_operand_end()};
  }

  const_operand_iterator true_operand_begin() const {
    return operand_begin() + getTrueDestOperandIndex();
  }
  const_operand_iterator true_operand_end() const {
    return true_operand_begin() + getNumTrueOperands();
  }
  llvm::iterator_range<const_operand_iterator> getTrueOperands() const {
    return {true_operand_begin(), true_operand_end()};
  }

  unsigned getNumTrueOperands() const;

  /// Erase the operand at 'index' from the true operand list.
  void eraseTrueOperand(unsigned index);

  // Accessors for operands to the 'false' destination.
  Value *getFalseOperand(unsigned idx) {
    assert(idx < getNumFalseOperands());
    return getOperand(getFalseDestOperandIndex() + idx);
  }
  const Value *getFalseOperand(unsigned idx) const {
    return const_cast<CondBranchOp *>(this)->getFalseOperand(idx);
  }
  void setFalseOperand(unsigned idx, Value *value) {
    assert(idx < getNumFalseOperands());
    setOperand(getFalseDestOperandIndex() + idx, value);
  }

  operand_iterator false_operand_begin() { return true_operand_end(); }
  operand_iterator false_operand_end() {
    return false_operand_begin() + getNumFalseOperands();
  }
  llvm::iterator_range<operand_iterator> getFalseOperands() {
    return {false_operand_begin(), false_operand_end()};
  }

  const_operand_iterator false_operand_begin() const {
    return true_operand_end();
  }
  const_operand_iterator false_operand_end() const {
    return false_operand_begin() + getNumFalseOperands();
  }
  llvm::iterator_range<const_operand_iterator> getFalseOperands() const {
    return {false_operand_begin(), false_operand_end()};
  }

  unsigned getNumFalseOperands() const;

  /// Erase the operand at 'index' from the false operand list.
  void eraseFalseOperand(unsigned index);

private:
  /// Get the index of the first true destination operand.
  unsigned getTrueDestOperandIndex() const { return 1; }

  /// Get the index of the first false destination operand.
  unsigned getFalseDestOperandIndex() const {
    return getTrueDestOperandIndex() + getNumTrueOperands();
  }

  friend class OperationInst;
  explicit CondBranchOp(const OperationInst *state) : Op(state) {}
};

/// The "constant" operation requires a single attribute named "value".
/// It returns its value as an SSA value.  For example:
///
///   %1 = "constant"(){value: 42} : i32
///   %2 = "constant"(){value: @foo} : (f32)->f32
///
class ConstantOp : public Op<ConstantOp, OpTrait::ZeroOperands,
                             OpTrait::OneResult, OpTrait::HasNoSideEffect> {
public:
  /// Builds a constant op with the specified attribute value and result type.
  static void build(Builder *builder, OperationState *result, Attribute value,
                    Type type);

  Attribute getValue() const { return getAttr("value"); }

  static StringRef getOperationName() { return "constant"; }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;
  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const;

protected:
  friend class OperationInst;
  explicit ConstantOp(const OperationInst *state) : Op(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning a float value of FloatType.
///
///   %1 = "constant"(){value: 42.0} : bf16
///
class ConstantFloatOp : public ConstantOp {
public:
  /// Builds a constant float op producing a float of the specified type.
  static void build(Builder *builder, OperationState *result,
                    const APFloat &value, FloatType type);

  APFloat getValue() const {
    return getAttrOfType<FloatAttr>("value").getValue();
  }

  static bool isClassFor(const OperationInst *op);

private:
  friend class OperationInst;
  explicit ConstantFloatOp(const OperationInst *state) : ConstantOp(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of IntegerType.
///
///   %1 = "constant"(){value: 42} : i32
///
class ConstantIntOp : public ConstantOp {
public:
  /// Build a constant int op producing an integer of the specified width.
  static void build(Builder *builder, OperationState *result, int64_t value,
                    unsigned width);

  /// Build a constant int op producing an integer with the specified type,
  /// which must be an integer type.
  static void build(Builder *builder, OperationState *result, int64_t value,
                    Type type);

  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value").getInt();
  }

  static bool isClassFor(const OperationInst *op);

private:
  friend class OperationInst;
  explicit ConstantIntOp(const OperationInst *state) : ConstantOp(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of Index type.
///
///   %1 = "constant"(){value: 99} : () -> index
///
class ConstantIndexOp : public ConstantOp {
public:
  /// Build a constant int op producing an index.
  static void build(Builder *builder, OperationState *result, int64_t value);

  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value").getInt();
  }

  static bool isClassFor(const OperationInst *op);

private:
  friend class OperationInst;
  explicit ConstantIndexOp(const OperationInst *state) : ConstantOp(state) {}
};

/// The "return" operation represents a return instruction within a function.
/// The operation takes variable number of operands and produces no results.
/// The operand number and types must match the signature of the function
/// that contains the operation. For example:
///
///   mlfunc @foo() : (i32, f8) {
///   ...
///   return %0, %1 : i32, f8
///
class ReturnOp : public Op<ReturnOp, OpTrait::VariadicOperands,
                           OpTrait::ZeroResult, OpTrait::IsTerminator> {
public:
  static StringRef getOperationName() { return "return"; }

  static void build(Builder *builder, OperationState *result,
                    ArrayRef<Value *> results = {});

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class OperationInst;
  explicit ReturnOp(const OperationInst *state) : Op(state) {}
};

// Prints dimension and symbol list.
void printDimAndSymbolList(OperationInst::const_operand_iterator begin,
                           OperationInst::const_operand_iterator end,
                           unsigned numDims, OpAsmPrinter *p);

// Parses dimension and symbol list and returns true if parsing failed.
bool parseDimAndSymbolList(OpAsmParser *parser,
                           SmallVector<Value *, 4> &operands,
                           unsigned &numDims);

void canonicalizeMapAndOperands(AffineMap &map,
                                llvm::SmallVectorImpl<Value *> &operands);

} // end namespace mlir

#endif
