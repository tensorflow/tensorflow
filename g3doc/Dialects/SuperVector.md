# SuperVector Dialect

This dialect provides mid-level abstraction for the MLIR super-vectorizer.

[TOC]

## Operations {#operations}

### Vector transfers {#vector-transfers}

#### `vector_transfer_read` operation {#'vector_transfer_read'-operation}

Syntax:

``` {.ebnf}
operation ::= ssa-id `=` `vector_transfer_read` ssa-use-list `{` attribute-entry `} :` function-type
```

Examples:

```mlir {.mlir}
// Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into vector<32x256xf32> and
// pad with %f0 to handle the boundary case:
%f0 = constant 0.0f : f32
for %i0 = 0 to %0 {
  for %i1 = 0 to %1 step 256 {
    for %i2 = 0 to %2 step 32 {
      %v = vector_transfer_read %A, %i0, %i1, %i2, %f0
           {permutation_map: (d0, d1, d2) -> (d2, d1)} :
           (memref<?x?x?xf32>, index, index, f32) -> vector<32x256xf32>
}}}

// Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into
// vector<128xf32>. The underlying implementation will require a 1-D vector
// broadcast:
for %i0 = 0 to %0 {
  for %i1 = 0 to %1 {
    %3 = vector_transfer_read %A, %i0, %i1
         {permutation_map: (d0, d1) -> (0)} :
         (memref<?x?xf32>, index, index) -> vector<128xf32>
  }
}
```

The `vector_transfer_read` performs a blocking read from a slice within a scalar
[MemRef](#memref-type) supplied as its first operand into a
[vector](#vector-type) of the same elemental type. The slice is further defined
by a full-rank index within the MemRef, supplied as the operands `2 .. 1 +
rank(memref)`. The permutation_map [attribute](#attributes) is an
[affine-map](#affine-maps) which specifies the transposition on the slice to
match the vector shape. The size of the slice is specified by the size of the
vector, given as the return type. Optionally, an `ssa-value` of the same
elemental type as the MemRef is provided as the last operand to specify padding
in the case of out-of-bounds accesses. Absence of the optional padding value
signifies the `vector_transfer_read` is statically guaranteed to remain within
the MemRef bounds. This operation is called 'read' by opposition to 'load'
because the super-vector granularity is generally not representable with a
single hardware register. A `vector_transfer_read` is thus a mid-level
abstraction that supports super-vectorization with non-effecting padding for
full-tile-only code.

More precisely, let's dive deeper into the permutation_map for the following :

```mlir {.mlir}
vector_transfer_read %A, %expr1, %expr2, %expr3, %expr4
  { permutation_map : (d0,d1,d2,d3) -> (d2,0,d0) } :
  (memref<?x?x?x?xf32>, index, index, index, index) -> vector<3x4x5xf32>
```

This operation always reads a slice starting at `%A[%expr1, %expr2, %expr3,
%expr4]`. The size of the slice is 3 along d2 and 5 along d0, so the slice is:
`%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

That slice needs to be read into a `vector<3x4x5xf32>`. Since the permutation
map is not full rank, there must be a broadcast along vector dimension `1`.

A notional lowering of vector_transfer_read could generate code resembling:

```mlir {.mlir}
// %expr1, %expr2, %expr3, %expr4 defined before this point
%tmp = alloc() : vector<3x4x5xf32>
%view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
for %i = 0 to 3 {
  for %j = 0 to 4 {
    for %k = 0 to 5 {
      %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
      store %tmp[%i, %j, %k] : vector<3x4x5xf32>
}}}
%c0 = constant 0 : index
%vec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
```

On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that the
temporary storage footprint is `3 * 5` values but `3 * 4 * 5` values are
actually transferred betwen `%A` and `%tmp`.

Alternatively, if a notional vector broadcast instruction were available, the
lowered code would resemble:

```mlir {.mlir}
// %expr1, %expr2, %expr3, %expr4 defined before this point
%tmp = alloc() : vector<3x4x5xf32>
%view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
for %i = 0 to 3 {
  for %k = 0 to 5 {
    %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
    store %tmp[%i, 0, %k] : vector<3x4x5xf32>
}}
%c0 = constant 0 : index
%tmpvec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
%vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
```

where `broadcast` broadcasts from element 0 to all others along the specified
dimension. This time, the temporary storage footprint is `3 * 5` values which is
the same amount of data as the `3 * 5` values transferred. An additional `1`
broadcast is required. On a GPU this broadcast could be implemented using a
warp-shuffle if loop `j` were mapped to `threadIdx.x`.

#### `vector_transfer_write` operation {#'vector_transfer_write'-operation}

Syntax:

``` {.ebnf}
operation ::= `vector_transfer_write` ssa-use-list `{` attribute-entry `} :` vector-type ', ' memref-type ', ' index-type-list
```

Examples:

```mlir {.mlir}
// write vector<16x32x64xf32> into the slice `%A[%i0, %i1:%i1+32, %i2:%i2+64, %i3:%i3+16]`:
for %i0 = 0 to %0 {
  for %i1 = 0 to %1 step 32 {
    for %i2 = 0 to %2 step 64 {
      for %i3 = 0 to %3 step 16 {
        %val = `ssa-value` : vector<16x32x64xf32>
        vector_transfer_write %val, %A, %i0, %i1, %i2, %i3
          {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
          vector<16x32x64xf32>, memref<?x?x?x?xf32>, index, index, index, index
}}}}
```

The `vector_transfer_write` performs a blocking write from a
[vector](#vector-type), supplied as its first operand, into a slice within a
scalar [MemRef](#memref-type) of the same elemental type, supplied as its second
operand. The slice is further defined by a full-rank index within the MemRef,
supplied as the operands `3 .. 2 + rank(memref)`. The permutation_map
[attribute](#attributes) is an [affine-map](#affine-maps) which specifies the
transposition on the slice to match the vector shape. The size of the slice is
specified by the size of the vector. This operation is called 'write' by
opposition to 'store' because the super-vector granularity is generally not
representable with a single hardware register. A `vector_transfer_write` is thus
a mid-level abstraction that supports super-vectorization with non-effecting
padding for full-tile-only code. It is the responsibility of
`vector_transfer_write`'s implementation to ensure the memory writes are valid.
Different lowerings may be pertinent depending on the hardware support.

### Vector views {#vector-views}

#### `vector_type_cast` operation {#'vector_type_cast'-operation}

Syntax:

``` {.ebnf}
operation ::= `vector_type_cast` ssa-use : memref-type, memref-type
```

Examples:

```mlir
 %A  = alloc() : memref<5x4x3xf32>
 %VA = vector_type_cast %A : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
```

The `vector_type_cast` operation performs a conversion from a memref with scalar
element to memref with a *single* vector element, copying the shape of the
memref to the vector. This is the minimal viable operation that is required to
make super-vectorization operational. It can be seen as a special case of the
`view` operation but scoped in the super-vectorization context.
