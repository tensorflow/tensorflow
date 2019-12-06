//===- VectorOps.td - Vector op definitions ---------------*- tablegen -*-====//
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
// Defines MLIR vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_OPS
#define VECTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/AffineOps/AffineOpsBase.td"

def Vector_Dialect : Dialect {
  let name = "vector";
  let cppNamespace = "vector";
}

// Base class for Vector dialect ops.
class Vector_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vector_Dialect, mnemonic, traits> {
  // For every vector op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// TODO(andydavis, ntv) Add an attribute to specify a different algebra
// with operators other than the current set: {*, +}.
def Vector_ContractionOp :
  Vector_Op<"contract", [NoSideEffect]>,
    Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc,
               Variadic<VectorOf<[I1]>>:$masks,
               AffineMapArrayAttr:$indexing_maps, ArrayAttr:$iterator_types)>,
    Results<(outs AnyVector)> {
  let summary = "vector contraction operation";
  let description = [{
    Computes the sum of products of vector elements along contracting
    dimension pairs from 2 vectors of rank M and N respectively, adds this
    intermediate result to the accumulator argument of rank K, and returns a
    vector result of rank K (where K = num_lhs_free_dims + num_rhs_free_dims +
    num_batch_dims (see dimension type descriptions below)).

    Optional vector mask arguments (produced by CreateMaskOp or ConstantMaskOp)
    specify the dynamic dimension sizes of valid data within the lhs/rhs vector
    arguments.

    An iterator type attribute list must be specified, where each element of
    the list represents an iterator with one of the following types:

    *) "reduction": reduction dimensions are present in the lhs and rhs
                    arguments but not in the output (or optional accumulator
                    argument). These are the dimensions along which the vector
                    contraction op computes the sum of products, and
                    contracting dimension pair dimension sizes must match
                    between lhs/rhs.
    *) "parallel": Batch dimensions are iterator type "parallel", and
                   are non-contracting dimensions present in the lhs, rhs and
                   output. The lhs/rhs co-iterate along the batch dimensions,
                   which should be expressed in their indexing maps.

                   Free dimensions are iterator type "parallel", and are
                   non-contraction, non-batch dimensions accessed by either the
                   lhs or rhs (but not both). The lhs and rhs free dimensions
                   are unrelated to each other and do not co-iterate, which
                   should be expressed in their indexing maps.

    An indexing map attribute list must be specified with an entry for lhs, rhs
    and acc arguments. An indexing map attribute specifies a mapping from each
    iterator in the iterator type list, to each dimension of an N-D vector.

    Examples:

      // 2D vector contraction with one contracting dimension (matmul).
      #contraction_accesses = [
        (i, j, k) -> (i, k),
        (i, j, k) -> (k, j),
        (i, j, k) -> (i, j)
      ]
      #contraction_trait = {
        indexing_maps = #contraction_accesses,
        iterator_types = [parallel, parallel, reduction]
      }

      %3 = vector.contract #contraction_trait %0, %1, %2
        : vector<4x3xf32>, vector<3x7xf32> into vector<4x7xf32>

      // 4D to 3D vector contraction with two contracting dimensions and
      // one batch dimension.
      #contraction_accesses = [
        (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
        (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
        (b0, f0, f1, c0, c1) -> (b0, f0, f1)
      ]
      #contraction_trait = {
        indexing_maps = #contraction_accesses,
        iterator_types = [parallel, parallel, parallel reduction, reduction]
      }

      %4 = vector.contract #contraction_trait %0, %1, %2
          : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>

      // 4D vector contraction with two contracting dimensions and optional
      // vector mask arguments.
      %lhs_mask = vector.constant_mask [7, 8, 16, 15] : vector<7x8x16x15xi1>
      %rhs_mask = vector.constant_mask [8, 16, 7, 5] : vector<8x16x7x5xi1>

      %5 = vector.contract #contraction_trait %0, %1, %2, %lhs_mask, %rhs_mask
         : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  }];
  let extraClassDeclaration = [{
    VectorType getLhsType() {
      return lhs()->getType().cast<VectorType>();
    }
    VectorType getRhsType() {
      return rhs()->getType().cast<VectorType>();
    }
    VectorType getAccType() {
      return acc()->getType().cast<VectorType>();
    }
    VectorType getLHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(3)->getType().cast<VectorType>();
    }
    VectorType getRHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(4)->getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult()->getType().cast<VectorType>();
    }
    SmallVector<StringRef, 2> getTraitAttrNames();
    SmallVector<AffineMap, 4> getIndexingMaps();
    static StringRef getReductionIteratorTypeName() {
      return "reduction";
    }
    static StringRef getParallelIteratorTypeName() {
      return "parallel";
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    // Returns a list of index maps, where there is a list entry for each
    // op indexing map attribute (i.e. one for each input and output, with
    // the output listed last). Each index map, maps from this operations
    // iteration space, to vector dimensions of the maps input/output.
    void getIterationIndexMap(
      std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap);

    std::vector<std::pair<int64_t, int64_t>> getContractingDimMap();
    std::vector<std::pair<int64_t, int64_t>> getBatchDimMap();
  }];
}

def Vector_BroadcastOp :
  Vector_Op<"broadcast", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyVector:$vector)> {
  let summary = "broadcast operation";
  let description = [{
    Broadcasts the scalar or k-D vector value in the source operand
    to a n-D result vector such that the broadcast makes sense, i.e.,
    the source operand is duplicated to match the given rank and sizes
    in the result vector. The legality rules are:
    * the source operand must have the same element type as the result type
    * a k-D vector <s_1 x .. x s_k x type> can be broadcast to
      a n-D vector <t_1 x .. x t_n x type> if
       * k <= n, and
       * the sizes in the trailing dimensions n-k < i <= n with j=i+k-n
          match exactly as s_j = t_i or s_j = 1:
       ```
           t_1 x   ..  t_n-k x t_n-k+1 x .. x t_i x .. x t_n
                               s_1     x .. x s_j x .. x s_k
               <duplication>         <potential stretch>
       ```
    The source operand is duplicated over all the missing leading dimensions
    and streched over the trailing dimensions where the source has a non-equal
    dimension of 1. These rules imply that any scalar broadcast (k=0) to any
    shaped vector with the same element type is always legal.

    Examples:
    ```
      %0 = constant 0.0 : f32
      %1 = vector.broadcast %0 : f32 to vector<16xf32>
      %2 = vector.broadcast %1 : vector<16xf32> to vector<4x16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    Type getSourceType() { return source()->getType(); }
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
  }];
}

def Vector_ExtractElementOp :
  Vector_Op<"extractelement", [NoSideEffect,
     PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I32ArrayAttr:$position)>,
    Results<(outs AnyType)> {
  let summary = "extractelement operation";
  let description = [{
    Takes an n-D vector and a k-D position and extracts the (n-k)-D vector at
    the proper position. Degenerates to an element type in the 0-D case.

    Examples:
    ```
      %1 = vector.extractelement %0[3]: vector<4x8x16xf32>
      %2 = vector.extractelement %0[3, 3, 3]: vector<4x8x16xf32>
    ```
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source, ArrayRef<int32_t>">];
  let extraClassDeclaration = [{
    static StringRef getPositionAttrName() { return "position"; }
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
  }];
}

def Vector_InsertElementOp :
  Vector_Op<"insertelement", [NoSideEffect,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     PredOpTrait<"dest operand and result have same type",
                 TCresIsSameAsOpBase<0, 1>>]>,
     Arguments<(ins AnyType:$source, AnyVector:$dest, I32ArrayAttr:$position)>,
     Results<(outs AnyVector)> {
  let summary = "insertelement operation";
  let description = [{
    Takes an n-D source vector, an (n+k)-D destination vector and a k-D position
    and inserts the n-D source into the (n+k)-D destination at the proper
    position. Degenerates to a scalar source type when n = 0.

    Examples:
    ```
      %2 = vector.insertelement %0, %1[3 : i32]:
        vector<8x16xf32> into vector<4x8x16xf32>
      %5 = vector.insertelement %3, %4[3 : i32, 3 : i32, 3 : i32]:
        f32 into vector<4x8x16xf32>
    ```
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source, " #
    "Value *dest, ArrayRef<int32_t>">];
  let extraClassDeclaration = [{
    static StringRef getPositionAttrName() { return "position"; }
    Type getSourceType() { return source()->getType(); }
    VectorType getDestVectorType() {
      return dest()->getType().cast<VectorType>();
    }
  }];
}

def Vector_InsertStridedSliceOp :
  Vector_Op<"insert_strided_slice", [NoSideEffect,
    PredOpTrait<"operand #0 and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
    PredOpTrait<"dest operand and result have same type",
                 TCresIsSameAsOpBase<0, 1>>]>,
    Arguments<(ins AnyVector:$source, AnyVector:$dest, I64ArrayAttr:$offsets,
               I64ArrayAttr:$strides)>,
    Results<(outs AnyVector)> {
  let summary = "strided_slice operation";
  let description = [{
    Takes a k-D source vector, an n-D destination vector (n >= k), n-D `offsets`
    integer array attribute, a k-D `strides` integer array attribute and inserts
    the k-D source vector as a strided subvector at the proper offset into the
    n-D destination vector.

    At the moment strides must contain only 1s.

    Returns an n-D vector that is a copy of the n-D destination vector in which
    the last k-D dimensions contain the k-D source vector elements strided at
    the proper location as specified by the offsets.

    Examples:
    ```
      %2 = vector.insert_strided_slice %0, %1
          {offsets : [0, 0, 2], strides : [1, 1]}:
        vector<2x4xf32> into vector<16x4x8xf32>
    ```
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source, Value *dest, " #
    "ArrayRef<int64_t> offsets, ArrayRef<int64_t> strides">];
  let extraClassDeclaration = [{
    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getSourceVectorType() {
      return source()->getType().cast<VectorType>();
    }
    VectorType getDestVectorType() {
      return dest()->getType().cast<VectorType>();
    }
  }];
}

def Vector_OuterProductOp :
  Vector_Op<"outerproduct", [NoSideEffect, SameOperandsAndResultElementType]>,
    Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, Variadic<AnyVector>:$acc)>,
    Results<(outs AnyVector)> {
  let summary = "vector outerproduct with optional fused add";
  let description = [{
    Takes 2 1-D vectors and returns the 2-D vector containing the outer product.

    An optional extra 2-D vector argument may be specified in which case the
    operation returns the sum of the outer product and the extra vector. When
    lowered to the LLVMIR dialect, this form emits `llvm.intr.fmuladd`, which
    can lower to actual `fma` instructions in LLVM.

    Examples

      %2 = vector.outerproduct %0, %1: vector<4xf32>, vector<8xf32>
      return %2: vector<4x8xf32>

      %3 = vector.outerproduct %0, %1, %2:
        vector<4xf32>, vector<8xf32>, vector<4x8xf32>
      return %3: vector<4x8xf32>
  }];
  let extraClassDeclaration = [{
    VectorType getOperandVectorTypeLHS() {
      return lhs()->getType().cast<VectorType>();
    }
    VectorType getOperandVectorTypeRHS() {
      return rhs()->getType().cast<VectorType>();
    }
    VectorType getOperandVectorTypeACC() {
      return (llvm::size(acc()) == 0) ? VectorType() :
        (*acc().begin())->getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return getResult()->getType().cast<VectorType>();
    }
  }];
}

def Vector_StridedSliceOp :
  Vector_Op<"strided_slice", [NoSideEffect,
    PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I64ArrayAttr:$offsets,
               I64ArrayAttr:$sizes, I64ArrayAttr:$strides)>,
    Results<(outs AnyVector)> {
  let summary = "strided_slice operation";
  let description = [{
    Takes an n-D vector, k-D `offsets` integer array attribute, a k-D `sizes`
    integer array attribute, a k-D `strides` integer array attribute and
    extracts the n-D subvector at the proper offset.

    At the moment strides must contain only 1s.
    // TODO(ntv) support non-1 strides.

    Returns an n-D vector where the first k-D dimensions match the `sizes`
    attribute. The returned subvector contains the elements starting at offset
    `offsets` and ending at `offsets + sizes`.

    Examples:
    ```
      %1 = vector.strided_slice %0
          {offsets : [0, 2], sizes : [2, 4], strides : [1, 1]}:
        vector<4x8x16xf32> to vector<2x4x16xf32>
    ```

    // TODO(ntv) Evolve to a range form syntax similar to:
    %1 = vector.strided_slice %0[0:2:1][2:4:1]
      vector<4x8x16xf32> to vector<2x4x16xf32>
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source, " #
    "ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, " #
    "ArrayRef<int64_t> strides">];
  let extraClassDeclaration = [{
    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getSizesAttrName() { return "sizes"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getVectorType(){ return vector()->getType().cast<VectorType>(); }
  }];
  let hasCanonicalizer = 1;
}

def Vector_TransferReadOp :
  Vector_Op<"transfer_read">,
    Arguments<(ins AnyMemRef:$memref, Variadic<Index>:$indices,
               AffineMapAttr:$permutation_map, AnyType:$padding)>,
    Results<(outs AnyVector:$vector)> {

  let summary = "Reads a supervector from memory into an SSA vector value.";

  let description = [{
    The `vector.transfer_read` op performs a blocking read from a slice within
    a scalar [MemRef](../LangRef.md#memref-type) supplied as its first operand
    into a [vector](../LangRef.md#vector-type) of the same elemental type. The
    slice is further defined by a full-rank index within the MemRef, supplied as
    the operands `2 .. 1 + rank(memref)`. The permutation_map
    [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The size of the slice is specified by the
    size of the vector, given as the return type. An `ssa-value` of the same
    elemental type as the MemRef is provided as the last operand to specify
    padding in the case of out-of-bounds accesses. This operation is called
    'read' by opposition to 'load' because the super-vector granularity is
    generally not representable with a single hardware register.
    A `vector.transfer_read` is thus a mid-level
    abstraction that supports super-vectorization with non-effecting padding for
    full-tile-only code.

    More precisely, let's dive deeper into the permutation_map for the following
    MLIR:

    ```mlir {.mlir}
    vector.transfer_read %A[%expr1, %expr2, %expr3, %expr4]
      { permutation_map : (d0,d1,d2,d3) -> (d2,0,d0) } :
      memref<?x?x?x?xf32>, vector<3x4x5xf32>
    ```

    This operation always reads a slice starting at `%A[%expr1, %expr2, %expr3,
    %expr4]`. The size of the slice is 3 along d2 and 5 along d0, so the slice
    is: `%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

    That slice needs to be read into a `vector<3x4x5xf32>`. Since the
    permutation map is not full rank, there must be a broadcast along vector
    dimension `1`.

    A notional lowering of vector.transfer_read could generate code resembling:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    %tmp = alloc() : vector<3x4x5xf32>
    %view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 5 {
          %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] :
            memref<?x?x?x?xf32>
          store %tmp[%i, %j, %k] : vector<3x4x5xf32>
    }}}
    %c0 = constant 0 : index
    %vec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
    ```

    On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that
    the temporary storage footprint is `3 * 5` values but `3 * 4 * 5` values are
    actually transferred between `%A` and `%tmp`.

    Alternatively, if a notional vector broadcast operation were available, the
    lowered code would resemble:

    ```mlir
    // %expr1, %expr2, %expr3, %expr4 defined before this point
    %tmp = alloc() : vector<3x4x5xf32>
    %view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
    for %i = 0 to 3 {
      affine.for %k = 0 to 5 {
        %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] :
          memref<?x?x?x?xf32>
        store %tmp[%i, 0, %k] : vector<3x4x5xf32>
    }}
    %c0 = constant 0 : index
    %tmpvec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
    %vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
    ```

    where `broadcast` broadcasts from element 0 to all others along the
    specified dimension. This time, the temporary storage footprint is `3 * 5`
    values which is the same amount of data as the `3 * 5` values transferred.
    An additional `1` broadcast is required. On a GPU this broadcast could be
    implemented using a warp-shuffle if loop `j` were mapped to `threadIdx.x`.

    Syntax
    ``` {.ebnf}
    operation ::= ssa-id `=` `vector.transfer_read` ssa-use-list
      `{` attribute-entry `} :` memref-type `,` vector-type
    ```

    Examples:

    ```mlir
    // Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into vector<32x256xf32>
    // and pad with %f0 to handle the boundary case:
    %f0 = constant 0.0f : f32
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 256 {
        affine.for %i2 = 0 to %2 step 32 {
          %v = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
               memref<?x?x?xf32>, vector<32x256xf32>
    }}}

    // Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into
    // vector<128xf32>. The underlying implementation will require a 1-D vector
    // broadcast:
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 {
        %3 = vector.transfer_read %A[%i0, %i1]
             {permutation_map: (d0, d1) -> (0)} :
             memref<?x?xf32>, vector<128xf32>
      }
    }
    ```
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref()->getType().cast<MemRefType>();
    }
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
  }];
}

def Vector_TransferWriteOp :
  Vector_Op<"transfer_write">,
    Arguments<(ins AnyVector:$vector, AnyMemRef:$memref,
               Variadic<Index>:$indices,
               AffineMapAttr:$permutation_map)> {

  let summary = "The vector.transfer_write op writes a supervector to memory.";

  let description = [{
    The `vector.transfer_write` performs a blocking write from a
    [vector](../LangRef.md#vector-type), supplied as its first operand, into a
    slice within a scalar [MemRef](../LangRef.md#memref-type) of the same
    elemental type, supplied as its second operand. The slice is further defined
    by a full-rank index within the MemRef, supplied as the operands
    `3 .. 2 + rank(memref)`.
    The permutation_map [attribute](../LangRef.md#attributes) is an
    [affine-map](Affine.md#affine-maps) which specifies the transposition on the
    slice to match the vector shape. The size of the slice is specified by the
    size of the vector. This operation is called 'write' by opposition to
    'store' because the super-vector granularity is generally not representable
    with a single hardware register. A `vector.transfer_write` is thus a
    mid-level abstraction that supports super-vectorization with non-effecting
    padding for full-tile-only code. It is the responsibility of
    `vector.transfer_write`'s implementation to ensure the memory writes are
    valid. Different lowerings may be pertinent depending on the hardware
    support.

    Syntax:

    ``` {.ebnf}
    operation ::= `vector.transfer_write` ssa-use-list `{` attribute-entry `} :
      ` vector-type ', ' memref-type '
    ```

    Examples:

    ```mlir {.mlir}
    // write vector<16x32x64xf32> into the slice
    //   `%A[%i0, %i1:%i1+32, %i2:%i2+64, %i3:%i3+16]`:
    for %i0 = 0 to %0 {
      affine.for %i1 = 0 to %1 step 32 {
        affine.for %i2 = 0 to %2 step 64 {
          affine.for %i3 = 0 to %3 step 16 {
            %val = `ssa-value` : vector<16x32x64xf32>
            vector.transfer_write %val, %A[%i0, %i1, %i2, %i3]
              {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
              vector<16x32x64xf32>, memref<?x?x?x?xf32>
    }}}}
    ```
  }];

  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
    MemRefType getMemRefType() {
      return memref()->getType().cast<MemRefType>();
    }
  }];
}

def Vector_TypeCastOp :
  Vector_Op<"type_cast", [NoSideEffect]>,
    Arguments<(ins StaticShapeMemRefOf<[AnyType]>:$memref)>,
    Results<(outs AnyMemRef)> {
  let summary = "type_cast op converts a scalar memref to a vector memref";
  let description = [{
    Performs a conversion from a memref with scalar element to a memref with a
    *single* vector element, copying the shape of the memref to the vector. This
    is the minimal viable operation that is required to makeke
    super-vectorization operational. It can be seen as a special case of the
    `view` operation but scoped in the super-vectorization context.

    Syntax:

    ``` {.ebnf}
    operation ::= `vector.type_cast` ssa-use : memref-type to memref-type
    ```

    Example:

    ```mlir
    %A  = alloc() : memref<5x4x3xf32>
    %VA = vector.type_cast %A : memref<5x4x3xf32> to memref<vector<5x4x3xf32>>
    ```
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source">];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref()->getType().cast<MemRefType>();
    }
    MemRefType getResultMemRefType() {
      return getResult()->getType().cast<MemRefType>();
    }
  }];
}

def Vector_ConstantMaskOp :
  Vector_Op<"constant_mask", [NoSideEffect]>,
    Arguments<(ins I64ArrayAttr:$mask_dim_sizes)>,
    Results<(outs VectorOf<[I1]>)> {
  let summary = "creates a constant vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the 'mask_dim_sizes'
    array attribute argument. Each element of the 'mask_dim_sizes' array,
    specifices an exclusive upper bound [0, mask-dim-size-element-value)
    for a unique dimension in the vector result. The conjunction of the ranges
    define a hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0).

    Example: create a constant vector mask of size 4x3xi1 with elements in range
             0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).

      %1 = vector.constant_mask [3, 2] : vector<4x3xi1>

      print %1
                    columns
                  0    1    2
                |------------
              0 | 1    1    0
        rows  1 | 1    1    0
              2 | 1    1    0
              3 | 0    0    0
  }];

  let extraClassDeclaration = [{
    static StringRef getMaskDimSizesAttrName() { return "mask_dim_sizes"; }
  }];
}

def Vector_CreateMaskOp :
  Vector_Op<"create_mask", [NoSideEffect]>,
    Arguments<(ins Variadic<Index>:$operands)>, Results<(outs VectorOf<[I1]>)> {
  let summary = "creates a vector mask";
  let description = [{
    Creates and returns a vector mask where elements of the result vector
    are set to '0' or '1', based on whether the element indices are contained
    within a hyper-rectangular region specified by the operands. Specifically,
    each operand specifies a range [0, operand-value) for a unique dimension in
    the vector result. The conjunction of the operand ranges define a
    hyper-rectangular region within which elements values are set to 1
    (otherwise element values are set to 0).

    Example: create a vector mask of size 4x3xi1 where elements in range
             0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).

      %1 = vector.create_mask %c3, %c2 : vector<4x3xi1>

      print %1
                    columns
                  0    1    2
                |------------
              0 | 1    1    0
        rows  1 | 1    1    0
              2 | 1    1    0
              3 | 0    0    0
  }];

  let hasCanonicalizer = 1;
}

#endif // VECTOR_OPS
