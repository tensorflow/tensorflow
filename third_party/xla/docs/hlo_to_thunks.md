# From HLO to Thunks

## Pre-optimization HLO
We start with pre-optimization HLO. Pre-optimization HLO does not contain ops
that are considered internal to XLA, e.g. `fusion` or `bitcast`. Ops don't have
a layout at this stage, or if they have, it will be ignored. Pre-optimization
HLO is usually produced by higher level frameworks like Tensorflow and JAX.
When using the XLA flag `-xla_dump_to`, the pre-optimization HLO is dumped to a
file with file name suffix “before_optimizations.txt”.

## Optimize HLO Module

The XLA:GPU pipeline will turn the pre-optimization HLO into optimized HLO by
running a sequence of passes. The passes can be grouped together semantically
and run in the following order:

### Sharding related passes

[Shardy Partitioner](https://openxla.org/shardy/overview) or SPMD sharding.

### Optimization passes.

This can include both legalization passes and simplification passes.

### Collective optimization passes.

Similar to **Optimization passes**, but focuses on collective ops.

### Layout assignment passes

Each HLO op is assigned a layout which is a part of the instruction shape. The
layout controls how the tensor is laid out physically in memory.

Example of a shape with layout:

```
f32[10,20,30]{2,0,1}
```

After the element type, there are the logical dimensions of the shape, followed
by the layout permutation in minor to major order. In this example, the most
minor dimension is 30, the second most minor dimension is 10, and the major
dimension is 20.

The goal of the layout assignment is to minimize the number of physical
transpositions that are required using a greedy strategy. It starts off with
certain layout constraints (e.g. CuDNN/cuBLAS libraries expect consecutive
dimensions) and propagates layout “down”, and then “up” the HLO graph. At the
end of layout propagation, some instructions may have conflicting layouts, one
propagated from an operand, one propagated from a user. To resolve this
conflict, a `copy` HLO instruction is inserted that changes the layout from the
operand layout to the instruction layout.

### Layout normalization passes

Given that it is somewhat difficult to figure out the physical shape, layout
normalization attempts to rewrite the shape such that it uses the default layout
`{rank-1, rank-2, …, 0}`. In the example above, the normalized shape would be
`f32[20,10,30]{2,1,0}`. Copy ops that change layouts are rewritten to a
combination of `transpose` + `bitcast`. Given that currently we cannot normalize
all ops, there are still some ops that may have non-default layouts, most
notably `gather` and `dot`. At the boundaries between normalized ops and
non-normalized ops there will be `bitcast` ops that represent a transpose, i.e.
a transpose with a layout assigned that makes it a no-op physically.

Layout normalization also makes some implicit transposes explicit which is
important because codegen can handle explicit transposes with a dedicated
emitter. For example, a reshape is technically allowed to have a different
physical layout between operand and result (e.g. due to different rank). The
`ReshapeDecomposer` pass that runs as part of the layout normalization passes
turns a reshape into a sequence of `transpose`, reshape `bitcast` and
`transpose`.

### Post layout assignment optimization passes

The most important passes here are Triton fusions (GEMM fusions +
Softmax/Layernorm fusions) or rewrites to library calls. But also Autotuning
runs in this step, where we pick the best algorithm for convolutions or dots, or
the best tiling for dots handled by the legacy Triton GEMM emitter, or whether
we should use Triton or Cublas for a certain dot fusion.

### Fusion passes

The two main passes are `PriorityFusion` and `Multi-Output` fusion.

In `PriorityFusion`, we form fusions guided by the cost model. When fusing we
would allow duplicating ops with several users if the op can be fused into all
users. We would also allow extending existing Triton Softmax fusions if
possible.

`Multi-Output` fusion is a separate pass that allows to fuse ops/fusions
together that share an operand, or fuse operands/operand fusions into users
without duplication but instead adding extra output(s) so other users of the op
to be fused can be redirected to this output. This pass needs to be careful not
to introduce cycles into the HLO graph.

 After Multi-Output fusion, we run common subexpression elimination (`HloCSE`
 pass) which will potentially merge previously duplicated ops back together if
 they ended up in the same fusion.

### Several post-fusion passes

Several passes related to collectives (like turning them to async, or enforcing
a certain relative order of collectives).

Finally we run `CopyInsertion` where copies are added to ensure that in-place
operations don't overwrite data that is still needed elsewhere.

At the end of optimization, the optimized HLO is dumped if using the flag
`-xla_dump_to` to a file that has the file name suffix
"after_optimizations.txt". If you want to dump the HLO after intermediate
passes that actually change the HloModule, you can use the flag
`-xla_dump_hlo_pass_re=.*` (or a specific regular expression to restrict it to
certain passes).

## Scheduling
An HloModule without schedule still has some degree of freedom in which order
the ops are processed. Basically any topological sort according to
operand/result relationship and control dependencies is ok. The scheduling
enforces a certain order. This influences the amount of memory that is required,
because we cannot reuse a buffer as long as not all readers of that buffer have
been processed. In an initial step, we try different scheduler algorithms and
pick the schedule that minimizes peak memory consumption.

As a follow-up, we run the `LatencyHidingScheduler` pass that tries to maximize
compute-communication overlap but may increase memory usage again.

After scheduling, we run `HloRematerialization` which attempts to reduce memory
usage in case peak memory consumption is higher than the amount of memory we
have available. This is at the cost of performance, as e.g. some fusions might
be split and some ops might be duplicated to have shorter buffer lifetimes. If
rematerialization is happening, it would potentially make sense to look if there
are ways at model side to reduce the amount of memory required (e.g. smaller
batch sizes).

## Thunks and CommandBuffers

TBD

## BufferAssignment

Immediately before we lower to LLVM IR, we run the buffer assignment passes that
will assign buffer slices to each instruction in the HLO graph. The buffer
assignment runs in several steps.

1. `HloDataflowAnalysis` assigns `HloValues` (essentially logical buffers) to
instructions. For in-place ops, the `HloValue` of an operand can be reused. An
op may define more than one `HloValue` (e.g. with a tuple result shape).

2. `HloAliasAnalysis` attempts to combine buffers for aliasing operations, and
computes a mapping from `HloValue` to `HloBuffer`.

3. `BufferAssignment` computes a mapping of `HloBuffers` to buffer slices inside
a big buffer in such a way that the same buffer slice is not used for different
`HloBuffers` with overlapping life times. For ops that may alias, it is ok that
there is a slight overlap (the end time of the one `HloBuffer` may coincide with
the start time of the other `HloBuffer`). When using the flag `-xla_dump_to`,
some information about buffer assignment is dumped to a file with the name
suffix "after_optimizations-buffer-assignment.txt".

