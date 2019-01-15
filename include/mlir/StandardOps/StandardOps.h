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

#ifndef MLIR_STANDARDOPS_STANDARDOPS_H
#define MLIR_STANDARDOPS_STANDARDOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
class AffineMap;
class Builder;

class StandardOpsDialect : public Dialect {
public:
  StandardOpsDialect(MLIRContext *context);
};

#define GET_OP_CLASSES
#include "mlir/StandardOps/standard_ops.inc"

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
  /// The result of an alloc is always a MemRefType.
  MemRefType getType() const {
    return getResult()->getType().cast<MemRefType>();
  }

  static StringRef getOperationName() { return "alloc"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result,
                    MemRefType memrefType, ArrayRef<Value *> operands = {});
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

private:
  friend class OperationInst;
  explicit AllocOp(const OperationInst *state) : Op(state) {}
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
                    ArrayRef<Value *> operands);

  Function *getCallee() const {
    return getAttrOfType<FunctionAttr>("callee").getValue();
  }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

protected:
  friend class OperationInst;
  explicit CallOp(const OperationInst *state) : Op(state) {}
};

/// The "call_indirect" operation represents an indirect call to a value of
/// function type.  Functions are first class types in MLIR, and may be passed
/// as arguments and merged together with block arguments.  The operands
/// and result types of the call must match the specified function type.
///
///   %31 = call_indirect %15(%0, %1)
///            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
///
class CallIndirectOp : public Op<CallIndirectOp, OpTrait::VariadicOperands,
                                 OpTrait::VariadicResults> {
public:
  static StringRef getOperationName() { return "call_indirect"; }

  static void build(Builder *builder, OperationState *result, Value *callee,
                    ArrayRef<Value *> operands);

  const Value *getCallee() const { return getOperand(0); }
  Value *getCallee() { return getOperand(0); }

  // Hooks to customize behavior of this op.
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

protected:
  friend class OperationInst;
  explicit CallIndirectOp(const OperationInst *state) : Op(state) {}
};

/// The predicate indicates the type of the comparison to perform:
/// (in)equality; (un)signed less/greater than (or equal to).
enum class CmpIPredicate {
  FirstValidValue,
  // (In)equality comparisons.
  EQ = FirstValidValue,
  NE,
  // Signed comparisons.
  SLT,
  SLE,
  SGT,
  SGE,
  // Unsigned comparisons.
  ULT,
  ULE,
  UGT,
  UGE,
  // Number of predicates.
  NumPredicates
};

/// The "cmpi" operation compares its two operands according to the integer
/// comparison rules and the predicate specified by the respective attribute.
/// The predicate defines the type of comparison: (in)equality, (un)signed
/// less/greater than (or equal to).  The operands must have the same type, and
/// this type must be an integer type, a vector or a tensor thereof.  The result
/// is an i1, or a vector/tensor thereof having the same shape as the inputs.
/// Since integers are signless, the predicate also explicitly indicates
/// whether to interpret the operands as signed or unsigned integers for
/// less/greater than comparisons.  For the sake of readability by humans,
/// short-hand syntax for the instruction uses a string-typed attribute for the
/// predicate.  The value of this attribute corresponds to lower-cased name of
/// the predicate constant, e.g., "slt" means "signed less than".  The string
/// representation of the attribute is merely a syntactic sugar and is converted
/// to an integer attribute by the parser.
///
///   %r1 = cmpi "eq" %0, %1 : i32
///   %r2 = cmpi "slt" %0, %1 : tensor<42x42xi64>
///   %r3 = "cmpi"(%0, %1){predicate: 0} : (i8, i8) -> i1
class CmpIOp
    : public Op<CmpIOp, OpTrait::OperandsAreIntegerLike,
                OpTrait::SameTypeOperands, OpTrait::NOperands<2>::Impl,
                OpTrait::OneResult, OpTrait::ResultsAreBoolLike,
                OpTrait::SameOperandsAndResultShape, OpTrait::HasNoSideEffect> {
public:
  CmpIPredicate getPredicate() const {
    return (CmpIPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
        .getInt();
  }

  static StringRef getOperationName() { return "cmpi"; }
  static StringRef getPredicateAttrName() { return "predicate"; }
  static CmpIPredicate getPredicateByName(StringRef name);

  static void build(Builder *builder, OperationState *result, CmpIPredicate,
                    Value *lhs, Value *rhs);
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;
  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const;

private:
  friend class OperationInst;
  explicit CmpIOp(const OperationInst *state) : Op(state) {}
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
  Value *getMemRef() { return getOperand(); }
  const Value *getMemRef() const { return getOperand(); }
  void setMemRef(Value *value) { setOperand(value); }

  static StringRef getOperationName() { return "dealloc"; }

  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result, Value *memref);
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

private:
  friend class OperationInst;
  explicit DeallocOp(const OperationInst *state) : Op(state) {}
};

/// The "dim" operation takes a memref or tensor operand and returns an
/// "index".  It requires a single integer attribute named "index".  It
/// returns the size of the specified dimension.  For example:
///
///   %1 = dim %0, 2 : tensor<?x?x?xf32>
///
class DimOp : public Op<DimOp, OpTrait::OneOperand, OpTrait::OneResult,
                        OpTrait::HasNoSideEffect> {
public:
  static void build(Builder *builder, OperationState *result,
                    Value *memrefOrTensor, unsigned index);

  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const;

  /// This returns the dimension number that the 'dim' is inspecting.
  unsigned getIndex() const {
    return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
  }

  static StringRef getOperationName() { return "dim"; }

  // Hooks to customize behavior of this op.
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class OperationInst;
  explicit DimOp(const OperationInst *state) : Op(state) {}
};

// DmaStartOp starts a non-blocking DMA operation that transfers data from a
// source memref to a destination memref. The source and destination memref need
// not be of the same dimensionality, but need to have the same elemental type.
// The operands include the source and destination memref's each followed by its
// indices, size of the data transfer in terms of the number of elements (of the
// elemental type of the memref), a tag memref with its indices, and optionally
// at the end, a stride and a number_of_elements_per_stride arguments. The tag
// location is used by a DmaWaitOp to check for completion. The indices of the
// source memref, destination memref, and the tag memref have the same
// restrictions as any load/store. The optional stride arguments should be of
// 'index' type, and specify a stride for the slower memory space (memory space
// with a lower memory space id), tranferring chunks of
// number_of_elements_per_stride every stride until %num_elements are
// transferred. Either both or no stride arguments should be specified.
//
// For example, a DmaStartOp operation that transfers 256 elements of a memref
// '%src' in memory space 0 at indices [%i, %j] to memref '%dst' in memory space
// 1 at indices [%k, %l], would be specified as follows:
//
//   %num_elements = constant 256
//   %idx = constant 0 : index
//   %tag = alloc() : memref<1 x i32, (d0) -> (d0), 4>
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
//     memref<40 x 128 x f32>, (d0) -> (d0), 0>,
//     memref<2 x 1024 x f32>, (d0) -> (d0), 1>,
//     memref<1 x i32>, (d0) -> (d0), 2>
//
//   If %stride and %num_elt_per_stride are specified, the DMA is expected to
//   transfer %num_elt_per_stride elements every %stride elements apart from
//   memory space 0 until %num_elements are transferred.
//
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
//             %num_elt_per_stride :
//
// TODO(mlir-team): add additional operands to allow source and destination
// striding, and multiple stride levels.
// TODO(andydavis) Consider replacing src/dst memref indices with view memrefs.
class DmaStartOp
    : public Op<DmaStartOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  static void build(Builder *builder, OperationState *result, Value *srcMemRef,
                    ArrayRef<Value *> srcIndices, Value *destMemRef,
                    ArrayRef<Value *> destIndices, Value *numElements,
                    Value *tagMemRef, ArrayRef<Value *> tagIndices,
                    Value *stride = nullptr,
                    Value *elementsPerStride = nullptr);

  // Returns the source MemRefType for this DMA operation.
  const Value *getSrcMemRef() const { return getOperand(0); }
  // Returns the rank (number of indices) of the source MemRefType.
  unsigned getSrcMemRefRank() const {
    return getSrcMemRef()->getType().cast<MemRefType>().getRank();
  }
  // Returns the source memerf indices for this DMA operation.
  llvm::iterator_range<OperationInst::const_operand_iterator>
  getSrcIndices() const {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_begin() + 1 + getSrcMemRefRank()};
  }

  // Returns the destination MemRefType for this DMA operations.
  const Value *getDstMemRef() const {
    return getOperand(1 + getSrcMemRefRank());
  }
  // Returns the rank (number of indices) of the destination MemRefType.
  unsigned getDstMemRefRank() const {
    return getDstMemRef()->getType().cast<MemRefType>().getRank();
  }
  unsigned getSrcMemorySpace() const {
    return getSrcMemRef()->getType().cast<MemRefType>().getMemorySpace();
  }
  unsigned getDstMemorySpace() const {
    return getDstMemRef()->getType().cast<MemRefType>().getMemorySpace();
  }

  // Returns the destination memref indices for this DMA operation.
  llvm::iterator_range<OperationInst::const_operand_iterator>
  getDstIndices() const {
    return {getInstruction()->operand_begin() + 1 + getSrcMemRefRank() + 1,
            getInstruction()->operand_begin() + 1 + getSrcMemRefRank() + 1 +
                getDstMemRefRank()};
  }

  // Returns the number of elements being transferred by this DMA operation.
  const Value *getNumElements() const {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank());
  }

  // Returns the Tag MemRef for this DMA operation.
  const Value *getTagMemRef() const {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1);
  }
  // Returns the rank (number of indices) of the tag MemRefType.
  unsigned getTagMemRefRank() const {
    return getTagMemRef()->getType().cast<MemRefType>().getRank();
  }

  // Returns the tag memref index for this DMA operation.
  llvm::iterator_range<OperationInst::const_operand_iterator>
  getTagIndices() const {
    unsigned tagIndexStartPos =
        1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1 + 1;
    return {getInstruction()->operand_begin() + tagIndexStartPos,
            getInstruction()->operand_begin() + tagIndexStartPos +
                getTagMemRefRank()};
  }

  /// Returns true if this is a DMA from a faster memory space to a slower one.
  bool isDestMemorySpaceFaster() const {
    return (getSrcMemorySpace() < getDstMemorySpace());
  }

  /// Returns true if this is a DMA from a slower memory space to a faster one.
  bool isSrcMemorySpaceFaster() const {
    // Assumes that a lower number is for a slower memory space.
    return (getDstMemorySpace() < getSrcMemorySpace());
  }

  /// Given a DMA start operation, returns the operand position of either the
  /// source or destination memref depending on the one that is at the higher
  /// level of the memory hierarchy. Asserts failure if neither is true.
  unsigned getFasterMemPos() const {
    assert(isSrcMemorySpaceFaster() || isDestMemorySpaceFaster());
    return isSrcMemorySpaceFaster() ? 0 : getSrcMemRefRank() + 1;
  }

  static StringRef getOperationName() { return "dma_start"; }
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

  bool isStrided() const {
    return getNumOperands() != 1 + getSrcMemRefRank() + 1 + getDstMemRefRank() +
                                   1 + 1 + getTagMemRefRank();
  }

  Value *getStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1 - 1);
  }
  const Value *getStride() const {
    return const_cast<DmaStartOp *>(this)->getStride();
  }

  Value *getNumElementsPerStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1);
  }
  const Value *getNumElementsPerStride() const {
    return const_cast<DmaStartOp *>(this)->getNumElementsPerStride();
  }

protected:
  friend class OperationInst;
  explicit DmaStartOp(const OperationInst *state) : Op(state) {}
};

// DmaWaitOp blocks until the completion of a DMA operation associated with the
// tag element '%tag[%index]'. %tag is a memref, and %index has to be an index
// with the same restrictions as any load/store index. %num_elements is the
// number of elements associated with the DMA operation. For example:
//
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
//     memref<2048 x f32>, (d0) -> (d0), 0>,
//     memref<256 x f32>, (d0) -> (d0), 1>
//     memref<1 x i32>, (d0) -> (d0), 2>
//   ...
//   ...
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 2>
//
class DmaWaitOp
    : public Op<DmaWaitOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  static void build(Builder *builder, OperationState *result, Value *tagMemRef,
                    ArrayRef<Value *> tagIndices, Value *numElements);

  static StringRef getOperationName() { return "dma_wait"; }

  // Returns the Tag MemRef associated with the DMA operation being waited on.
  const Value *getTagMemRef() const { return getOperand(0); }
  Value *getTagMemRef() { return getOperand(0); }

  // Returns the tag memref index for this DMA operation.
  llvm::iterator_range<OperationInst::const_operand_iterator>
  getTagIndices() const {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_begin() + 1 + getTagMemRefRank()};
  }

  // Returns the rank (number of indices) of the tag memref.
  unsigned getTagMemRefRank() const {
    return getTagMemRef()->getType().cast<MemRefType>().getRank();
  }

  // Returns the number of elements transferred in the associated DMA operation.
  const Value *getNumElements() const {
    return getOperand(1 + getTagMemRefRank());
  }

  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

protected:
  friend class OperationInst;
  explicit DmaWaitOp(const OperationInst *state) : Op(state) {}
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
class ExtractElementOp
    : public Op<ExtractElementOp, OpTrait::VariadicOperands, OpTrait::OneResult,
                OpTrait::HasNoSideEffect> {
public:
  static void build(Builder *builder, OperationState *result, Value *aggregate,
                    ArrayRef<Value *> indices = {});

  Value *getAggregate() { return getOperand(0); }
  const Value *getAggregate() const { return getOperand(0); }

  llvm::iterator_range<OperationInst::operand_iterator> getIndices() {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_end()};
  }

  llvm::iterator_range<OperationInst::const_operand_iterator>
  getIndices() const {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_end()};
  }

  static StringRef getOperationName() { return "extract_element"; }

  // Hooks to customize behavior of this op.
  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class OperationInst;
  explicit ExtractElementOp(const OperationInst *state) : Op(state) {}
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
  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result, Value *memref,
                    ArrayRef<Value *> indices = {});

  Value *getMemRef() { return getOperand(0); }
  const Value *getMemRef() const { return getOperand(0); }
  void setMemRef(Value *value) { setOperand(0, value); }
  MemRefType getMemRefType() const {
    return getMemRef()->getType().cast<MemRefType>();
  }

  llvm::iterator_range<OperationInst::operand_iterator> getIndices() {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_end()};
  }

  llvm::iterator_range<OperationInst::const_operand_iterator>
  getIndices() const {
    return {getInstruction()->operand_begin() + 1,
            getInstruction()->operand_end()};
  }

  static StringRef getOperationName() { return "load"; }

  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

private:
  friend class OperationInst;
  explicit LoadOp(const OperationInst *state) : Op(state) {}
};

/// The "memref_cast" operation converts a memref from one type to an equivalent
/// type with a compatible shape.  The source and destination types are
/// when both are memref types with the same element type, affine mappings,
/// address space, and rank but where the individual dimensions may add or
/// remove constant dimensions from the memref type.
///
/// If the cast converts any dimensions from an unknown to a known size, then it
/// acts as an assertion that fails at runtime of the dynamic dimensions
/// disagree with resultant destination size.
///
/// Assert that the input dynamic shape matches the destination static shape.
///    %2 = memref_cast %1 : memref<?x?xf32> to memref<4x4xf32>
/// Erase static shape information, replacing it with dynamic information.
///    %3 = memref_cast %1 : memref<4xf32> to memref<?xf32>
///
class MemRefCastOp : public CastOp<MemRefCastOp> {
public:
  static StringRef getOperationName() { return "memref_cast"; }

  /// The result of a memref_cast is always a memref.
  MemRefType getType() const {
    return getResult()->getType().cast<MemRefType>();
  }

  bool verify() const;

private:
  friend class OperationInst;
  explicit MemRefCastOp(const OperationInst *state) : CastOp(state) {}
};

/// The "select" operation chooses one value based on a binary condition
/// supplied as its first operand. If the value of the first operand is 1, the
/// second operand is chosen, otherwise the third operand is chosen. The second
/// and the third operand must have the same type. The operation applies
/// elementwise to vectors and tensors.  The shape of all arguments must be
/// identical. For example, the maximum operation is obtained by combining
/// "select" with "cmpi" as follows.
///
///   %2 = cmpi "gt" %0, %1 : i32         // %2 is i1
///   %3 = select %2, %0, %1 : i32
///
class SelectOp : public Op<SelectOp, OpTrait::NOperands<3>::Impl,
                           OpTrait::OneResult, OpTrait::HasNoSideEffect> {
public:
  static StringRef getOperationName() { return "select"; }
  static void build(Builder *builder, OperationState *result, Value *condition,
                    Value *trueValue, Value *falseValue);
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;
  bool verify() const;

  Value *getCondition() { return getOperand(0); }
  const Value *getCondition() const { return getOperand(0); }
  Value *getTrueValue() { return getOperand(1); }
  const Value *getTrueValue() const { return getOperand(1); }
  Value *getFalseValue() { return getOperand(2); }
  const Value *getFalseValue() const { return getOperand(2); }

  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const;

private:
  friend class OperationInst;
  explicit SelectOp(const OperationInst *state) : Op(state) {}
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
  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result,
                    Value *valueToStore, Value *memref,
                    ArrayRef<Value *> indices = {});

  Value *getValueToStore() { return getOperand(0); }
  const Value *getValueToStore() const { return getOperand(0); }

  Value *getMemRef() { return getOperand(1); }
  const Value *getMemRef() const { return getOperand(1); }
  void setMemRef(Value *value) { setOperand(1, value); }
  MemRefType getMemRefType() const {
    return getMemRef()->getType().cast<MemRefType>();
  }

  llvm::iterator_range<OperationInst::operand_iterator> getIndices() {
    return {getInstruction()->operand_begin() + 2,
            getInstruction()->operand_end()};
  }

  llvm::iterator_range<OperationInst::const_operand_iterator>
  getIndices() const {
    return {getInstruction()->operand_begin() + 2,
            getInstruction()->operand_end()};
  }

  static StringRef getOperationName() { return "store"; }

  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

private:
  friend class OperationInst;
  explicit StoreOp(const OperationInst *state) : Op(state) {}
};

/// The "tensor_cast" operation converts a tensor from one type to an equivalent
/// type without changing any data elements.  The source and destination types
/// must both be tensor types with the same element type, and the source and
/// destination types may not be the same.  They must either have the same rank,
/// or one may be an unknown rank.  The operation is invalid if converting to a
/// mismatching constant dimension.
///
/// Convert from unknown rank to rank 2 with unknown dimension sizes.
///    %2 = tensor_cast %1 : tensor<??f32> to tensor<?x?xf32>
///
class TensorCastOp : public CastOp<TensorCastOp> {
public:
  static StringRef getOperationName() { return "tensor_cast"; }

  /// The result of a tensor_cast is always a tensor.
  TensorType getType() const {
    return getResult()->getType().cast<TensorType>();
  }

  bool verify() const;

private:
  friend class OperationInst;
  explicit TensorCastOp(const OperationInst *state) : CastOp(state) {}
};

} // end namespace mlir

#endif
