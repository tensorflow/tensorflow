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

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif // AFFINE_OPS_BASE

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

def VectorExtractElementOp :
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

def VectorOuterProductOp :
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

def VectorTransferReadOp :
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

def VectorTransferWriteOp :
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

def VectorTypeCastOp :
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
#endif // VECTOR_OPS
