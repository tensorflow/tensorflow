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
class MLValue;

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
                    ArrayRef<SSAValue *> operands);

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

private:
  friend class Operation;
  explicit AffineApplyOp(const Operation *state) : Op(state) {}
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
                    Type *type);

  Attribute getValue() const { return getAttr("value"); }

  static StringRef getOperationName() { return "constant"; }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;
  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const;

protected:
  friend class Operation;
  explicit ConstantOp(const Operation *state) : Op(state) {}
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
                    const APFloat &value, FloatType *type);

  APFloat getValue() const {
    return getAttrOfType<FloatAttr>("value").getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantFloatOp(const Operation *state) : ConstantOp(state) {}
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
                    Type *type);

  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value").getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantIntOp(const Operation *state) : ConstantOp(state) {}
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
    return getAttrOfType<IntegerAttr>("value").getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantIndexOp(const Operation *state) : ConstantOp(state) {}
};

/// The "return" operation represents a return statement of an ML function.
/// The operation takes variable number of operands and produces no results.
/// The operand number and types must match the signature of the ML function
/// that contains the operation. For example:
///
///   mlfunc @foo() : (i32, f8) {
///   ...
///   return %0, %1 : i32, f8
///
class ReturnOp
    : public Op<ReturnOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  static StringRef getOperationName() { return "return"; }

  static void build(Builder *builder, OperationState *result,
                    ArrayRef<SSAValue *> results);

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class Operation;
  explicit ReturnOp(const Operation *state) : Op(state) {}
};

// Prints dimension and symbol list.
void printDimAndSymbolList(Operation::const_operand_iterator begin,
                           Operation::const_operand_iterator end,
                           unsigned numDims, OpAsmPrinter *p);

// Parses dimension and symbol list and returns true if parsing failed.
bool parseDimAndSymbolList(OpAsmParser *parser,
                           SmallVector<SSAValue *, 4> &operands,
                           unsigned &numDims);

} // end namespace mlir

#endif
