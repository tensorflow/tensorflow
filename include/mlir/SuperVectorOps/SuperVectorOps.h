//===- SuperVectorOps.h - MLIR Super Vectorizer Operations ------*- C++ -*-===//
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
// This file defines convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INCLUDE_MLIR_SUPERVECTOROPS_SUPERVECTOROPS_H
#define MLIR_INCLUDE_MLIR_SUPERVECTOROPS_SUPERVECTOROPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Dialect for super-vectorization Ops.
class SuperVectorOpsDialect : public Dialect {
public:
  SuperVectorOpsDialect(MLIRContext *context);
};

/// VectorTransferReadOp performs a blocking read from a scalar memref
/// location into a super-vector of the same elemental type. This operation is
/// called 'read' by opposition to 'load' because the super-vector granularity
/// is generally not representable with a single hardware register. As a
/// consequence, memory transfers will generally be required when lowering
/// VectorTransferReadOp. A VectorTransferReadOp is thus a mid-level abstraction
/// that supports super-vectorization with non-effecting padding for full-tile
/// only code.
//
/// A vector transfer read has semantics similar to a vector load, with
/// additional support for:
///   1. an optional value of the elemental type of the MemRef. This value
///      supports non-effecting padding and is inserted in places where the
///      vector read exceeds the MemRef bounds. If the value is not specified,
///      the access is statically guaranteed to be within bounds;
///   2. an attribute of type AffineMap to specify a slice of the original
///      MemRef access and its transposition into the super-vector shape.
///      The permutation_map is an unbounded AffineMap that must
///      represent a permutation from the MemRef dim space projected onto the
///      vector dim space.
///      This permutation_map has as many output dimensions as the vector rank.
///      However, it is not necessarily full rank on the target space to signify
///      that broadcast operations will be needed along certain vector
///      dimensions.
///      In the limit, one may load a 0-D slice of a memref (i.e. a single
///      value) into a vector, which corresponds to broadcasting that value in
///      the whole vector (i.e. a non-constant splat).
///
/// Example with full rank permutation_map:
/// ```mlir
///   %A = alloc(%size1, %size2, %size3, %size4) : memref<?x?x?x?xf32>
///   ...
///   %val = `ssa-value` : f32
///   // let %i, %j, %k, %l be ssa-values of type index
///   %v0 = vector_transfer_read %src, %i, %j, %k, %l
///          {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
///        (memref<?x?x?x?xf32>, index, index, index, index) ->
///          vector<16x32x64xf32>
///   %v1 = vector_transfer_read %src, %i, %j, %k, %l, %val
///          {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
///        (memref<?x?x?x?xf32>, index, index, index, index, f32) ->
///           vector<16x32x64xf32>
/// ```
///
/// Example with partial rank permutation_map:
/// ```mlir
///   %c0 = constant 0 : index
///   %A = alloc(%size1, %size2, %size3, %size4) : memref<?x?x?x?xf32>
///   ...
///   // let %i, %j be ssa-values of type index
///   %v0 = vector_transfer_read %src, %i, %c0, %c0, %c0
///          {permutation_map: (d0, d1, d2, d3) -> (0, d1, 0)} :
///        (memref<?x?x?x?xf32>, index, index, index, index) ->
///          vector<16x32x64xf32>
class VectorTransferReadOp
    : public Op<VectorTransferReadOp, OpTrait::VariadicOperands,
                OpTrait::OneResult> {
  enum Offsets : unsigned { MemRefOffset = 0, FirstIndexOffset = 1 };

public:
  static StringRef getOperationName() { return "vector_transfer_read"; }
  static StringRef getPermutationMapAttrName() { return "permutation_map"; }
  static void build(Builder *builder, OperationState *result,
                    VectorType vectorType, SSAValue *srcMemRef,
                    ArrayRef<SSAValue *> srcIndices, AffineMap permutationMap,
                    Optional<SSAValue *> paddingValue = None);
  VectorType getResultType() const {
    return getResult()->getType().cast<VectorType>();
  }
  SSAValue *getVector() { return getResult(); }
  const SSAValue *getVector() const { return getResult(); }
  SSAValue *getMemRef() { return getOperand(Offsets::MemRefOffset); }
  const SSAValue *getMemRef() const {
    return getOperand(Offsets::MemRefOffset);
  }
  VectorType getVectorType() const { return getResultType(); }
  MemRefType getMemRefType() const {
    return getMemRef()->getType().cast<MemRefType>();
  }
  llvm::iterator_range<Operation::operand_iterator> getIndices();
  llvm::iterator_range<Operation::const_operand_iterator> getIndices() const;
  Optional<SSAValue *> getPaddingValue();
  Optional<const SSAValue *> getPaddingValue() const;
  AffineMap getPermutationMap() const;

  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class Operation;
  explicit VectorTransferReadOp(const Operation *state) : Op(state) {}
};

/// VectorTransferWriteOp performs a blocking write from a super-vector to
/// a scalar memref of the same elemental type. This operation is
/// called 'write' by opposition to 'store' because the super-vector granularity
/// is generally not representable with a single hardware register. As a
/// consequence, memory transfers will generally be required when lowering
/// VectorTransferWriteOp. A VectorTransferWriteOp is thus a mid-level
/// abstraction that supports super-vectorization with non-effecting padding for
/// full-tile only code.
///
/// A vector transfer write has semantics similar to a vector store, with
/// additional support for handling out-of-bounds situations. It is the
/// responsibility of vector_transfer_write's implementation to ensure the
/// memory writes are valid. Different implementations may be pertinent
/// depending on the hardware support including:
/// 1. predication;
/// 2. explicit control-flow;
/// 3. Read-Modify-Write;
/// 4. writing out of bounds of the memref when the allocation allows it.
///
/// Example:
/// ```mlir
///   %A = alloc(%size1, %size2, %size3, %size4) : memref<?x?x?x?xf32>.
///   %val = `ssa-value` : vector<16x32x64xf32>
///   // let %i, %j, %k, %l be ssa-values of type index
///   vector_transfer_write %val, %src, %i, %j, %k, %l
///     {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
///   vector<16x32x64xf32>, memref<?x?x?x?xf32>, index, index, index, index
/// ```
class VectorTransferWriteOp
    : public Op<VectorTransferWriteOp, OpTrait::VariadicOperands,
                OpTrait::ZeroResult> {
  enum Offsets : unsigned {
    VectorOffset = 0,
    MemRefOffset = 1,
    FirstIndexOffset = 2
  };

public:
  static StringRef getOperationName() { return "vector_transfer_write"; }
  static StringRef getPermutationMapAttrName() { return "permutation_map"; }
  static void build(Builder *builder, OperationState *result,
                    SSAValue *srcVector, SSAValue *dstMemRef,
                    ArrayRef<SSAValue *> dstIndices, AffineMap permutationMap);
  SSAValue *getVector() { return getOperand(Offsets::VectorOffset); }
  const SSAValue *getVector() const {
    return getOperand(Offsets::VectorOffset);
  }
  VectorType getVectorType() const {
    return getVector()->getType().cast<VectorType>();
  }
  SSAValue *getMemRef() { return getOperand(Offsets::MemRefOffset); }
  const SSAValue *getMemRef() const {
    return getOperand(Offsets::MemRefOffset);
  }
  MemRefType getMemRefType() const {
    return getMemRef()->getType().cast<MemRefType>();
  }
  llvm::iterator_range<Operation::operand_iterator> getIndices();
  llvm::iterator_range<Operation::const_operand_iterator> getIndices() const;
  AffineMap getPermutationMap() const;

  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class Operation;
  explicit VectorTransferWriteOp(const Operation *state) : Op(state) {}
};

/// VectorTypeCastOp performs a conversion from a memref with scalar element to
/// memref with vector element, copying the shape of the memref to the vector.
///
/// Example:
///
/// ```mlir
///  %A  = alloc() : memref<5x4x3xf32>
///  %VA = vector_type_cast %A : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
/// ```
class VectorTypeCastOp
    : public Op<VectorTypeCastOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  static StringRef getOperationName() { return "vector_type_cast"; }
  static void build(Builder *builder, OperationState *result,
                    SSAValue *srcVector, Type dstType);
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

private:
  friend class Operation;
  explicit VectorTypeCastOp(const Operation *state) : Op(state) {}
};

} // end namespace mlir

#endif // MLIR_INCLUDE_MLIR_SUPERVECTOROPS_SUPERVECTOROPS_H
