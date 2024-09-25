# XLA MLIR fusion emitters

This is a prototype of a new loop emitter. The main goals are:

- Fixing exponential code size issues with the current emitter. We should be
  able to generate reasonable code for any fusion (note that execution time may
  still be bad, but that's a problem for priority fusion).
- Fixing compile time (as a result of the above).
- Make the code easier to understand thanks to gradual lowering.
- Eventually extend the concepts here to the other emitters (transpose, reduce
  in particular)

## High-level overview

The code consists of the following big building blocks:

- Computation partitioning - splitting an HLO computation into functions
- Elemental emission of XLA instructions
- Based on the above two: emission of functions
- The actual emitter
- Lowerings to LLVM

## Partitioning

See `computation_partitioner.h`.

Non-elementwise HLO instructions cannot always be emitted together. Consider the
following HLO graph:

```
     param
       |
      log
      |  \
      |  transpose
      |  /
      add
```

If we emit this in a single function, the `log` will be accessed at two
different indices for each element of the `add`. The old emitters solve this
problem by generating the `log` twice. For this particular graph, this is not
a problem, but when there are multiple splits, the code size grows
exponentially.

Here, we solve this problem by partitioning the graph into pieces that can be
safely emitted as one function. The criteria are:

- Instructions that have only one user are safe to emit together with their
  user.
- Instructions that have multiple users are safe to emit together with their
  users if they are accessed through the same indices by all users.

In the example above, the `add` and `tranpose` access different indices of the
`log`, so it is not safe to emit it together with them.

The graph is therefore partitioned into three functions (each containing just
one instruction).

## Elemental emission

See `elemental_hlo_to_mlir.h`.

Elemental emission is based on `mlir_hlo` and reuses it for all element-wise
instructions. For the most part, this is straightforward, but there are some
interesting things going on here.

### Indexing transformations

Some instructions (`transpose`, `broadcast`, `reshape`, `slice`, `reverse` and
a few more) are purely transformations on indices: to produce an element of the
result, we need to produce some other element of the input. For this, we can
reuse XLA's `indexing_analysis`, which has functions to produce the output to
input mapping for an instruction.

For example, for a `transpose` from `[20,40]` to `[40,20]`, it will produce the
following indexing map (one affine expression per input dimension; d0 and d1 are
the output dimensions):

```
  (d0, d1) -> d1
  (d0, d1) -> d0
```

So for these pure index transformation instructions, we can simply get the map,
apply it to the output indices, and produce the input at the resulting index.

Similarly, the `pad` op uses indexing maps and constraints for most of the
implementation. `pad` is also an indexing transformation with some added checks
to see if we return an element of the input or the padding value.

### Tuples

We do not support internal `tuple`s. We also do not support nested tuple
outputs. All XLA graphs that use these features can be converted to graphs that
do not.

### Gather

We only support canonical gathers as produced by [`gather_simplifier`](
https://github.com/openxla/xla/blob/main/xla/service/gather_simplifier.h).

## Emission of functions

For a subgraph of a computation with parameters `%p0` to `%p_n`, and subgraph
roots with rank `r` and element types (`e0` to `e_m`), we use the following MLIR
function signature:

``````
(%p0: tensor<...>, %p1: tensor<...>, ..., %pn: tensor<...>,
 %i0: index, %i1: index, ..., %i_r-1: index) -> (e0, ..., e_m)
``````

That is, we have one tensor input per computation parameter, one index input per
dimension of the output, and one result per output.

To emit a function, we simply use the elemental emitter above, and recursively
emit its operands until we reach the edge of the subgraph. Then, we:

- emit a `tensor.extract` for parameters
- or emit a `func.call` for other subgraphs

## Putting it together: the loop emitter

The loop emitter first partitions its fusion computation and emits code for each
subgraph. Then, it has to generate an entry function. The entry function is
different from the functions above, since it has no indices as inputs (just the
thread and block IDs) and actually needs to write the output somewhere. For the
loop emitter, this is fairly straightforward, but the transpose and reduction
emitters have non-trivial write logic.

The signature of the entry computation is:

```
(%p0: tensor<...>, ..., %pn: tensor<...>,
 %r0: tensor<...>, ..., %rn: tensor<...>) -> (tensor<...>, ..., tensor<...>)
```

Where like before, the `%pn`s are the parameters of the computation, and the
`%rn`s are the results of the computation. The entry computation takes the
results as tensors, `tensor.insert`s updates into them, and then returns them.
No other uses of the output tensors are allowed.

## Lowerings to LLVM

We mostly use the standard LLVM lowerings, but there are a few special passes.
We cannot use the `memref` lowerings for tensors, since we don't bufferize the
IR and our ABI is not compatible with the `memref` ABI. Instead, we have a
custom lowering directly from tensors to `LLVM`.

- The lowering of tensors is done in `lower_tensors.cc`. `tensor.extract` is
  lowered to `llvm.load`, `tensor.insert` to `llvm.store`, in the obvious way.
- `propagate_slice_indices` and `merge_pointers_to_same_slice` together
  implement a detail of buffer assignment and XLA's ABI: if two tensors share
  the same buffer slice, they are only passed once. These passes deduplicate the
  function arguments.

