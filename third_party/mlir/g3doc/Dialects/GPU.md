# GPU Dialect

Note: this dialect is more likely to change than others in the near future; use
with caution.

This dialect provides middle-level abstractions for launching GPU kernels
following a programming model similar to that of CUDA or OpenCL. It provides
abstractions for kernel invocations (and may eventually provide those for device
management) that are not present at the lower level (e.g., as LLVM IR intrinsics
for GPUs). Its goal is to abstract away device- and driver-specific
manipulations to launch a GPU kernel and provide a simple path towards GPU
execution from MLIR. It may be targeted, for example, by DSLs using MLIR. The
dialect uses `gpu` as its canonical prefix.

## Memory attribution

Memory buffers are defined at the function level, either in "gpu.launch" or in
"gpu.func" ops. This encoding makes it clear where the memory belongs and makes
the lifetime of the memory visible. The memory is only accessible while the
kernel is launched/the function is currently invoked. The latter is more strict
than actual GPU implementations but using static memory at the function level is
just for convenience. It is also always possible to pass pointers to the
workgroup memory into other functions, provided they expect the correct memory
space.

The buffers are considered live throughout the execution of the GPU function
body. The absence of memory attribution syntax means that the function does not
require special buffers. Rationale: although the underlying models declare
memory buffers at the module level, we chose to do it at the function level to
provide some structuring for the lifetime of those buffers; this avoids the
incentive to use the buffers for communicating between different kernels or
launches of the same kernel, which should be done through function arguments
instead; we chose not to use `alloca`-style approach that would require more
complex lifetime analysis following the principles of MLIR that promote
structure and representing analysis results in the IR.

## Operations

### `gpu.block_dim`

Returns the number of threads in the thread block (aka the block size) along the
x, y, or z `dimension`.

Example:

```mlir
  %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
```

### `gpu.block_id`

Returns the block id, i.e. the index of the current block within the grid along
the x, y, or z `dimension`.

Example:

```mlir
  %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
```

### `gpu.grid_dim`

Returns the number of thread blocks in the grid along the x, y, or z
`dimension`.

Example:

```mlir
  %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)
```

### `gpu.thread_id`

Returns the thread id, i.e. the index of the current thread within the block
along the x, y, or z `dimension`.

Example:

```mlir
  %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
```

### `gpu.yield`

Is a special terminator operation for blocks inside regions in gpu ops. It
returns values to the immediately enclosing gpu op.

Example:

```mlir
gpu.yield %f0, %f1 : f32, f32
```

### `gpu.all_reduce`

The "all_reduce" op reduces the value of every work item across a local
workgroup. The result is equal for all work items of a workgroup.

For example, both

```mlir
%1 = "gpu.all_reduce"(%0) ({}) { op = "add" } : (f32) -> (f32)
%2 = "gpu.all_reduce"(%0) ({
^bb(%lhs : f32, %rhs : f32):
  %sum = addf %lhs, %rhs : f32
  "gpu.yield"(%sum) : (f32) -> ()
}) : (f32) -> (f32)
```

compute the sum of each work item's %0 value. The first version specifies the
accumulation as operation, whereas the second version specifies the accumulation
as code region. The accumulation operation must either be `add` or `mul`.

Either none or all work items of a workgroup need to execute this op
in convergence.

### `gpu.barrier`

The "barrier" op synchronizes all work items of a workgroup. It is used
to coordinate communication between the work items of the workgroup.

```mlir
gpu.barrier
```

waits until all work items in the workgroup have reached this point and all
memory accesses made by these work items prior to the op are visible to all work
items in the workgroup. Data hazards between work items accessing the same
memory can be avoided by synchronizing work items in-between these accesses.

Either none or all work items of a workgroup need to execute this op
in convergence.
