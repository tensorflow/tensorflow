# Google ML Structured Dialect
 
The `gml_st` dialect will contain a loop-like construct and subset operations
that should allow support for fusion beyond rectangular tiles. This is necessary
for operations like `gather`, `scatter`, `concat` and more.

## Overview
### Tiling and fusion

Tiling of an op is performed by creating a loop that computes subsets of the
result. Usually the tiling is needed to enable vectorization or distribution.

Before tiling

```
%0 = op(%input)
```

After tiling

```
loop (%ivs)
  %1 = subset(%input, %ivs)
  %2 = op (%1)
```

Fusion of a producer op into a tiled consumer consists of two main parts:
computing subsets of producer's operands and moving the producer op into the
loop body so that it operates on the subsets of its original operands.

After consumer tiling
```
%0 = producer (%input)
loop (%ivs)
  %1 = subset(%0, %ivs)
  %2 = consumer(%1)
```

After producer fusion

```
loop (%ivs)
  %0 = subset(%input, %ivs)
  %1 = producer(%0)
  %2 = consumer (%1)
```

There is some duality between tiling and fusion. One can consider tiling as
fusion of the op into a loop that partitions the iteration space and just
returns identity for every subset. On the other hand, fusion can be seen as
tiling of the producer and then merging of the loop bodies.

### Subset operations

Linalg has support for hyperrectangular subsets (tiles) of tensor/memref
operands. Currently, Linalg's fusion assumes that the tiling is performed only
using `tensor.extract_slice/tensor.insert_slice` and `memref.subview`
operations.
There are several disadvantages to that approach:

If some of the operands are not affected by tiling, i.e. the tiling was
performed along dimensions that are not present in the operand, then we cannot
fuse anymore the producer of the operand. That can happen when `linalg.generic`
broadcasts one of the operands or when the output is tiled, but not the
reduction dimensions

Support for fusion with ops like `gather`, `scatter`, `concat` for some of the
cases can only be done via `TilingInterface`
([RFC](https://llvm.discourse.group/t/rfc-for-tilinginterface-for-tiling-operations-that-dont-fit-into-linalg-structured-operation-definition/3897/7)).

**Example of a tiled op**

```
%sum = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
          ins (%in_ = %in: tensor<80x60xf32>, %cst_ = %cst: f32)
          outs (%out_ = %out: tensor<80xf32>)
          iterators["parallel", "reduction"] {
  %in_sub = tensor.extract_slice %in_[%i, %j] [4, 4] [1, 1]
      : tensor<80x60xf32> to tensor<4x4xf32>
  %out_sub = tensor.extract_slice %out_[%i] [4] [1]
      : tensor<80xf32> to tensor<4xf32>
  %reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%in_sub : tensor<4x4xf32>)
      outs(%out_sub : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -> tensor<4xf32>
  %update = tensor.insert_slice %reduction into %out_[%i] [4] [1]
      : tensor<4xf32> into tensor<80xf32>
  linalg.yield %update : tensor<80xf32>
}
```

The body of this loop models read-modify-write of the output tensor. The tile
that we extract from `%out_` should have the same sizes/offsets/strides as the
destination of `tensor.insert_slice`. The arguments of `tensor.extract_slice`
and `tensor.insert_slice` are currently not required to encode the same tile.

We introduce new operations that define subsets on tensors/memrefs

 * `subset.full %tensor` - the subset spans the original tensor fully
 * `subset.tile %tensor [%offsets][%sizes][%strides]` - defines a rectangular
   tile
 * `subset.filter %tensor[%indices]` - the subset has the same shape as the
   original tensor, but only the values at %indices are populated. This can be a
   sparse tensor.
 * `subset.point %tensor[%index]` - the subset contains a single element

### Structured loop

We introduce `gml_st.loop` that keeps the subset definition separately from the
materialization.

`linalg.generic` has `AffineMap` attributes that specify the indexing maps and a
region that models the computation on the element types of the operand
tensors/memrefs. The region ends with `linalg.yield` terminator that yields the
element of the output. The load and store ops in that case are implicit, so
are extraction/insertion in `gml_st.loop`.

`gml_st.loop` has one region that contains subset operations to define the
dense/sparse ranges that we are working with and also `gml_st.materialize` ops
to convert subset spec to a tensor or memref.

`gml_st.yield` is the terminator for `gml_st.loop` that takes computed tensors
and a subset specification for which the computation was done. Note that this
way we don't have to explicitly write a destructive update with
`tensor.insert_slice` and then yield a full tensor. Here, we yield values for a
subset.


```
%sum = gml_st.loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
           ins (%in_ = %in: tensor<80x60xf32>, %cst_ = %cst: f32)
           outs (%out_ = %out: tensor<80xf32>)
           iterators["parallel", "sequential"] {
  %in_tile = gml_st.tile %in_[%i, %j] [4, 4] [1, 1]
      : tensor<80x60xf32> to !gml_st.subset<4x4xf32>
  %out_tile = gml_st.tile %out_[%i] [4] [1]
      : tensor<80xf32> to !gml_st.subset<4xf32>

  %in_sub = gml_st.materialize %in_tile
      : !gml_st.subset<4x4xf32> to tensor<4x4xf32>
  %out_sub = gml_st.materialize %in_tile
      : !gml_st.subset<4xf32> to tensor<4xf32>
  %reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%in_sub : tensor<4x4xf32>)
      outs(%out_sub : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -> tensor<4xf32>
  gml_st.yield %reduction to %out_tile
      : tensor<4xf32> to !gml_st.subset<4xf32>
}
```

Currently, tiling of the consumer and fusion of its producers are tightly
coupled. If the fusion is happening not in the same pass, then some analysis is
required to find the [consumer - `tensor.extract_slice` - producer] triple to
perform the fusion. Keeping the subset computations separately from the
"compute" ops not only improves readability but also simplifies fusion, since we
have a subset computation per operand and we can just specify what argument of
the loop we want to fuse.

It also simplifies the bufferization, since we don't need to introduce the
additional operations in MemRef dialect for every subset operation in TensorOps.
