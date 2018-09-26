//===- StandardOps.h - Standard MLIR Operations -----------------*- C++ -*-===//
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
// This file defines convenience types for working with standard operations
// in the MLIR instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STANDARDOPS_H
#define MLIR_IR_STANDARDOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class OperationSet;
class Builder;

/// The "addf" operation takes two operands and returns one result, each of
/// these is required to be of the same type.  This type may be a floating point
/// scalar type, a vector whose element type is a floating point type, or a
/// floating point tensor. For example:
///
///   %2 = addf %0, %1 : f32
///
class AddFOp : public BinaryOp<AddFOp, OpTrait::ResultsAreFloatLike> {
public:
  static StringRef getOperationName() { return "addf"; }

  Attribute *constantFold(ArrayRef<Attribute *> operands,
                          MLIRContext *context) const;

private:
  friend class Operation;
  explicit AddFOp(const Operation *state) : BinaryOp(state) {}
};

/// The "affine_apply" operation applies an affine map to a list of operands,
/// yielding a list of results. The operand and result list sizes must be the
/// same. All operands and results are of type 'AffineInt'. This operation
/// requires a single affine map attribute named "map".
/// For example:
///
///   %y = "affine_apply" (%x) { map: (d0) -> (d0 + 1) } :
///          (affineint) -> (affineint)
///
/// equivalently:
///
///   #map42 = (d0)->(d0+1)
///   %y = affine_apply #map42(%x)
///
class AffineApplyOp : public Op<AffineApplyOp, OpTrait::VariadicOperands,
                                OpTrait::VariadicResults> {
public:
  /// Builds an affine apply op with the specified map and operands.
  static void build(Builder *builder, OperationState *result, AffineMap *map,
                    ArrayRef<SSAValue *> operands);

  /// Returns the affine map to be applied by this operation.
  AffineMap *getAffineMap() const {
    return getAttrOfType<AffineMapAttr>("map")->getValue();
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
  bool constantFold(ArrayRef<Attribute *> operands,
                    SmallVectorImpl<Attribute *> &results,
                    MLIRContext *context) const;

private:
  friend class Operation;
  explicit AffineApplyOp(const Operation *state) : Op(state) {}
};

/// The "alloc" operation allocates a region of memory, as specified by its
/// memref type. For example:
///
///   %0 = alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
///
/// The optional list of dimension operands are bound to the dynamic dimensions
/// specified in its memref type. In the example below, the ssa value '%d' is
/// bound to the second dimension of the memref (which is dynamic).
///
///   %0 = alloc(%d) : memref<8x?xf32, (d0, d1) -> (d0, d1), 1>
///
/// The optional list of symbol operands are bound to the symbols of the
/// memrefs affine map. In the example below, the ssa value '%s' is bound to
/// the symbol 's0' in the affine map specified in the allocs memref type.
///
///   %0 = alloc()[%s] : memref<8x64xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1>
///
/// This operation returns a single ssa value of memref type, which can be used
/// by subsequent load and store operations.
class AllocOp
    : public Op<AllocOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
public:
  SSAValue *getMemRef() { return getOperation()->getResult(0); }
  const SSAValue *getMemRef() const { return getOperation()->getResult(0); }

  static StringRef getOperationName() { return "alloc"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result,
                    MemRefType *memrefType, ArrayRef<SSAValue *> operands = {});
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit AllocOp(const Operation *state) : Op(state) {}
};

/// The "call" operation represents a direct call to a function.  The operands
/// and result types of the call must match the specified function type.  The
/// callee is encoded as a function attribute named "callee".
///
///   %31 = call @my_add(%0, %1)
///            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
class CallOp
    : public Op<CallOp, OpTrait::VariadicOperands, OpTrait::VariadicResults> {
public:
  static StringRef getOperationName() { return "call"; }

  static void build(Builder *builder, OperationState *result, Function *callee,
                    ArrayRef<SSAValue *> operands);

  Function *getCallee() const {
    return getAttrOfType<FunctionAttr>("callee")->getValue();
  }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

protected:
  friend class Operation;
  explicit CallOp(const Operation *state) : Op(state) {}
};

/// The "call_indirect" operation represents an indirect call to a value of
/// function type.  Functions are first class types in MLIR, and may be passed
/// as arguments and merged together with basic block arguments.  The operands
/// and result types of the call must match the specified function type.
///
///   %31 = call_indirect %15(%0, %1)
///            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
///
class CallIndirectOp : public Op<CallIndirectOp, OpTrait::VariadicOperands,
                                 OpTrait::VariadicResults> {
public:
  static StringRef getOperationName() { return "call_indirect"; }

  static void build(Builder *builder, OperationState *result, SSAValue *callee,
                    ArrayRef<SSAValue *> operands);

  const SSAValue *getCallee() const { return getOperand(0); }
  SSAValue *getCallee() { return getOperand(0); }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

protected:
  friend class Operation;
  explicit CallIndirectOp(const Operation *state) : Op(state) {}
};

/// The "constant" operation requires a single attribute named "value".
/// It returns its value as an SSA value.  For example:
///
///   %1 = "constant"(){value: 42} : i32
///   %2 = "constant"(){value: @foo} : (f32)->f32
///
class ConstantOp
    : public Op<ConstantOp, OpTrait::ZeroOperands, OpTrait::OneResult> {
public:
  /// Builds a constant op with the specified attribute value and result type.
  static void build(Builder *builder, OperationState *result, Attribute *value,
                    Type *type);

  Attribute *getValue() const { return getAttr("value"); }

  static StringRef getOperationName() { return "constant"; }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;
  Attribute *constantFold(ArrayRef<Attribute *> operands,
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
  static void build(Builder *builder, OperationState *result, double value,
                    FloatType *type);

  double getValue() const {
    return getAttrOfType<FloatAttr>("value")->getValue();
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

  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value")->getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantIntOp(const Operation *state) : ConstantOp(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of AffineInt type.
///
///   %1 = "constant"(){value: 99} : () -> affineint
///
class ConstantAffineIntOp : public ConstantOp {
public:
  /// Build a constant int op producing an affineint.
  static void build(Builder *builder, OperationState *result, int64_t value);

  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value")->getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantAffineIntOp(const Operation *state) : ConstantOp(state) {}
};

/// The "dealloc" operation frees the region of memory referenced by a memref
/// which was originally created by the "alloc" operation.
/// The "dealloc" operation should not be called on memrefs which alias an
//  alloc'd memref (i.e. memrefs returned by the "view" and "reshape"
/// operations).
///
///   %0 = alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
///
///   dealloc %0 : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
///
class DeallocOp
    : public Op<DeallocOp, OpTrait::OneOperand, OpTrait::ZeroResult> {
public:
  SSAValue *getMemRef() { return getOperand(); }
  const SSAValue *getMemRef() const { return getOperand(); }

  static StringRef getOperationName() { return "dealloc"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result, SSAValue *memref);
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit DeallocOp(const Operation *state) : Op(state) {}
};

/// The "dim" operation takes a memref or tensor operand and returns an
/// "affineint".  It requires a single integer attribute named "index".  It
/// returns the size of the specified dimension.  For example:
///
///   %1 = dim %0, 2 : tensor<?x?x?xf32>
///
class DimOp : public Op<DimOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  /// This returns the dimension number that the 'dim' is inspecting.
  unsigned getIndex() const {
    return (unsigned)getAttrOfType<IntegerAttr>("index")->getValue();
  }

  static StringRef getOperationName() { return "dim"; }

  // Hooks to customize behavior of this op.
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit DimOp(const Operation *state) : Op(state) {}
};

/// The "extract_element" op reads a tensor or vector and returns one element
/// from it specified by an index list. The output of extract is a new value
/// with the same type as the elements of the tensor or vector. The arity of
/// indices matches the rank of the accessed value (i.e., if a tensor is of rank
/// 3, then 3 indices are required for the extract).  The indices should all be
/// of affine_int type.
///
/// For example:
///
///   %3 = extract_element %0[%1, %2] : vector<4x4xi32>
///
class ExtractElementOp : public Op<ExtractElementOp, OpTrait::VariadicOperands,
                                   OpTrait::OneResult> {
public:
  static void build(Builder *builder, OperationState *result,
                    SSAValue *aggregate, ArrayRef<SSAValue *> indices = {});

  SSAValue *getAggregate() { return getOperand(0); }
  const SSAValue *getAggregate() const { return getOperand(0); }

  llvm::iterator_range<Operation::operand_iterator> getIndices() {
    return {getOperation()->operand_begin() + 1, getOperation()->operand_end()};
  }

  llvm::iterator_range<Operation::const_operand_iterator> getIndices() const {
    return {getOperation()->operand_begin() + 1, getOperation()->operand_end()};
  }

  static StringRef getOperationName() { return "extract_element"; }

  // Hooks to customize behavior of this op.
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit ExtractElementOp(const Operation *state) : Op(state) {}
};

/// The "load" op reads an element from a memref specified by an index list. The
/// output of load is a new value with the same type as the elements of the
/// memref. The arity of indices is the rank of the memref (i.e., if the memref
/// loaded from is of rank 3, then 3 indices are required for the load following
/// the memref identifier).  For example:
///
///   %3 = load %0[%1, %1] : memref<4x4xi32>
///
class LoadOp
    : public Op<LoadOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
public:
  SSAValue *getMemRef() { return getOperand(0); }
  const SSAValue *getMemRef() const { return getOperand(0); }

  llvm::iterator_range<Operation::operand_iterator> getIndices() {
    return {getOperation()->operand_begin() + 1, getOperation()->operand_end()};
  }

  llvm::iterator_range<Operation::const_operand_iterator> getIndices() const {
    return {getOperation()->operand_begin() + 1, getOperation()->operand_end()};
  }

  static StringRef getOperationName() { return "load"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result, SSAValue *memref,
                    ArrayRef<SSAValue *> indices = {});
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit LoadOp(const Operation *state) : Op(state) {}
};

/// The "mulf" operation takes two operands and returns one result, each of
/// these is required to be of the same type.  This type may be a floating point
/// scalar type, a vector whose element type is a floating point type, or a
/// floating point tensor. For example:
///
///   %2 = mulf %0, %1 : f32
///
class MulFOp : public BinaryOp<MulFOp, OpTrait::ResultsAreFloatLike> {
public:
  static StringRef getOperationName() { return "mulf"; }

  Attribute *constantFold(ArrayRef<Attribute *> operands,
                          MLIRContext *context) const;

private:
  friend class Operation;
  explicit MulFOp(const Operation *state) : BinaryOp(state) {}
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

/// The "shape_cast" operation converts a tensor from one type to an equivalent
/// type without changing any data elements.  The source and destination types
/// must both be tensor types with the same element type, and the source and
/// destination types may not be the same.  They must either have the same rank,
/// or one may be an unknown rank.  The operation is invalid if converting to a
/// mismatching constant dimension.
///
/// Convert from unknown rank to rank 2 with unknown dimension sizes.
///    %2 = shape_cast %1 : tensor<??f32> to tensor<?x?xf32>
///
class ShapeCastOp
    : public Op<ShapeCastOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  static StringRef getOperationName() { return "shape_cast"; }

  static void build(Builder *builder, OperationState *result, SSAValue *input,
                    Type *resultType);

  /// The result of a shape_cast is always a tensor.
  TensorType *getType() const {
    return cast<TensorType>(getResult()->getType());
  }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class Operation;
  explicit ShapeCastOp(const Operation *state) : Op(state) {}
};

/// The "store" op writes an element to a memref specified by an index list.
/// The arity of indices is the rank of the memref (i.e. if the memref being
/// stored to is of rank 3, then 3 indices are required for the store following
/// the memref identifier). The store instruction does not produce a result.
///
/// In the following example, the ssa value '%v' is stored in memref '%A' at
/// indices [%i, %j]:
///
///   store %v, %A[%i, %j] : memref<4x128xf32, (d0, d1) -> (d0, d1), 0>
///
class StoreOp
    : public Op<StoreOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  SSAValue *getValueToStore() { return getOperand(0); }
  const SSAValue *getValueToStore() const { return getOperand(0); }

  SSAValue *getMemRef() { return getOperand(1); }
  const SSAValue *getMemRef() const { return getOperand(1); }

  llvm::iterator_range<Operation::operand_iterator> getIndices() {
    return {getOperation()->operand_begin() + 2, getOperation()->operand_end()};
  }

  llvm::iterator_range<Operation::const_operand_iterator> getIndices() const {
    return {getOperation()->operand_begin() + 2, getOperation()->operand_end()};
  }

  static StringRef getOperationName() { return "store"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result,
                    SSAValue *valueToStore, SSAValue *memref,
                    ArrayRef<SSAValue *> indices = {});
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit StoreOp(const Operation *state) : Op(state) {}
};

/// Install the standard operations in the specified operation set.
void registerStandardOperations(OperationSet &opSet);

} // end namespace mlir

#endif
