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

/// The "addf" operation takes two operands and returns one result, each of
/// these is required to be of the same type.  This type may be a floating point
/// scalar type, a vector whose element type is a floating point type, or a
/// floating point tensor. For example:
///
///   %2 = addf %0, %1 : f32
///
class AddFOp
    : public OpBase<AddFOp, OpTrait::NOperands<2>::Impl, OpTrait::OneResult> {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "addf"; }

  const char *verify() const;
  static OpAsmParserResult parse(OpAsmParser *parser);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit AddFOp(const Operation *state) : OpBase(state) {}
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
class AffineApplyOp : public OpBase<AffineApplyOp, OpTrait::VariadicOperands,
                                    OpTrait::VariadicResults> {
public:
  // Returns the affine map to be applied by this operation.
  AffineMap *getAffineMap() const {
    return getAttrOfType<AffineMapAttr>("map")->getValue();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "affine_apply"; }

  // Hooks to customize behavior of this op.
  static OpAsmParserResult parse(OpAsmParser *parser);
  void print(OpAsmPrinter *p) const;
  const char *verify() const;

private:
  friend class Operation;
  explicit AffineApplyOp(const Operation *state) : OpBase(state) {}
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
    : public OpBase<AllocOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
public:
  SSAValue *getMemRef() { return getOperation()->getResult(0); }
  const SSAValue *getMemRef() const { return getOperation()->getResult(0); }

  static StringRef getOperationName() { return "alloc"; }

  // Hooks to customize behavior of this op.
  const char *verify() const;
  static OpAsmParserResult parse(OpAsmParser *parser);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit AllocOp(const Operation *state) : OpBase(state) {}
};

/// The "constant" operation requires a single attribute named "value".
/// It returns its value as an SSA value.  For example:
///
///   %1 = "constant"(){value: 42} : i32
///   %2 = "constant"(){value: @foo} : (f32)->f32
///
class ConstantOp
    : public OpBase<ConstantOp, OpTrait::ZeroOperands, OpTrait::OneResult/*,
                         OpTrait::HasAttributeBase<"foo">::Impl*/> {
public:
  Attribute *getValue() const { return getAttr("value"); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "constant"; }

  // Hooks to customize behavior of this op.
  const char *verify() const;

protected:
  friend class Operation;
  explicit ConstantOp(const Operation *state) : OpBase(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value (either an IntegerType or AffineInt).
///
///   %1 = "constant"(){value: 42}
///
class ConstantIntOp : public ConstantOp {
public:
  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value")->getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantIntOp(const Operation *state) : ConstantOp(state) {}
};

/// The "dim" operation takes a memref or tensor operand and returns an
/// "affineint".  It requires a single integer attribute named "index".  It
/// returns the size of the specified dimension.  For example:
///
///   %1 = dim %0, 2 : tensor<?x?x?xf32>
///
class DimOp : public OpBase<DimOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  /// This returns the dimension number that the 'dim' is inspecting.
  unsigned getIndex() const {
    return (unsigned)getAttrOfType<IntegerAttr>("index")->getValue();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "dim"; }

  // Hooks to customize behavior of this op.
  const char *verify() const;
  static OpAsmParserResult parse(OpAsmParser *parser);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit DimOp(const Operation *state) : OpBase(state) {}
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
    : public OpBase<LoadOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
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
  const char *verify() const;
  static OpAsmParserResult parse(OpAsmParser *parser);
  void print(OpAsmPrinter *p) const;

private:
  friend class Operation;
  explicit LoadOp(const Operation *state) : OpBase(state) {}
};

/// Install the standard operations in the specified operation set.
void registerStandardOperations(OperationSet &opSet);

} // end namespace mlir

#endif
