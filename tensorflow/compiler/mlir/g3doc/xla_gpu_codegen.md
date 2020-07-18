# MLIR CodeGen for XLA

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'timshen' reviewed: '2020-06-16' }
*-->

XLA operates on `HloInstruction` and performs many optimizations on this
representation, sharing a lot of these between targeted devices. As some point a
linear schedule is computed and the memory buffer is assigned to each value
statically. The device specific codegen operates by traversing this sequence and
calling "emitters" to generate a representation suitable for the device (for
example a single LLVM function per XLA computation on CPU, or a sequence of
"thunks" encapsulating GPU operations and possibly generated PTX when targeting
GPU).

As a staging step, we're currently in the process of intercepting the process
right after XLA completes the buffer-assignment phase and emit instead an MLIR
module in the `lhlo` dialect. From there we perform the codegen using MLIR
components (Linalg, affine, and GPU dialect mainly) depending on the device.

Below is the plan of record to incrementally migrate XLA/GPU by using `lhlo` as
the codegen input.

## Tasks

|               | Host                     | Device
| ------------- | ------------------------ | ------------------------
| Input format  | HloInstruction* (Task 1) | HloInstruction* (Task 1)
| Output format | xla::Thunk (Task 2)      | LLVM IR (Task 3)

*   **Task 1** changes both host and device input format from HloInstruction* to
    LHLO.
*   **Task 2** changes output format of host from thunks to "some landing pad
    for host" (see below).
*   **Task 3** migrates device output from LLVM IR to some form of MLIR. It's
    optional to this project, and see the section "Migrating Device LLVM IR" for
    details.

This project prioritizes having end-to-end runnable models with LHLO-emitters
enabled as much as possible. This implies that the following order list of
objectives by priority:

*   Make XLA/GPU runnable with LHLO emitters, with existing Thunks and emitters
    unmodified.
*   Eliminate the references to HloInstruction\* in LHLO, case by case:
    *   Switch a legacy emitter to an MLIR-based emitter (e.g. Linalg), or
    *   Mechanically translate the existing emitter to take MLIR representation
        (migrate to Standard with GPU Dialect).

## Migrating Thunks (Task 2)

xla::gpu::Thunk is a data structure that:

*   Can be called into from the host (xla::gpu::Thunk::ExecuteOnStream()).
*   Carries various data in its subclasses.
*   Interacts with BufferAllocation::Slice and StreamExecutor.
*   Launches kernels
*   Calls into all runtime libraries.

The cost of that includes:

*   Representing op-specific configuration data (e.g. convolution configs).
*   Migrating op shape and operand shapes.
*   Representing a tree of thunks (while, condition, etc).

The migration work is independent from LHLO / emitter migration. Under limited
resources, it's prioritized behind LHLO / emitter migration.

We have several choices on how to lower the host-side part from LHLO:

*   TFRT
    *   (Pro) great CUDA and HIP wrappers for use.
    *   (Pro) easy to implement library calls (cuDNN, cuBLAS, cuFFT, etc), as
        TFRT ops are interpreted by C++ code.
    *   (Con) host side is under development and not tested.
    *   (Con) the JAX integration isn’t clear from a runtime point of view
*   Jitted CPU code
    *   (Pro) great lower-ability. Create a few loops and conditions and it's
        done.
    *   (Con) GPUDialect doesn't yet model chains/streams/asynchronicity/device
        allocation.
    *   (Con) CUDA / HIP runtime support is minimal (toolkit path, version,
        dynamic loading, etc).
*   Existing (interpreting) XLA runtime

Tentative conclusion: Use jitted CPU code during the transition, and optionally
adopt TFRT in the end.

## Migrating Device LLVM IR (Task 3)

An elemental emitter generates target op by filling it element by element. Each
output element depends on a set of elements from the operands. All elements are
described by combining the buffer with dynamic indices. It's sufficient to
describe almost all "math" ops, but for performance reasons only a large subset
of "math" ops are implemented directly in (Cpu|Gpu)ElementalIrEmitter.

ElementalIrEmitter is unique in that:

*   A large portion of the code is shared between XLA/GPU and CPU.
*   It represents a large portion of ops seen in models, including all
    element-wise ops.
*   Most fusions solely depend on ElementalIrEmitter.
*   It's structurally simple, as it describes a data dependency DAG between op
    elements and operand elements.
*   It's mostly portable and high-level (e.g. unlike GPU kReduce and GPU kCopy).
*   Dynamic shape support is easy for at least element-wise ops.

Now, for all ops, elementally-emitted or not, there are several flavors of the
end state of each XLA op:

1.  Device code stays as LLVM IR.
1.  Refactor the old emitter to be like LHLO -> MLIR LLVM Dialect:
    *   (Cost) Will be throw-away work if we want to ultimately migrate to
        Standard.
    *   (Benefit) It is easy and mechanical. Can be done in a short period.
    *   (Benefit) It doesn't benefit more compared to a).
1.  Refactor old emitters to be like LHLO -> MLIR GPU + Standard + Loops:
    *   (Cost) Lifting existing emitters to Standard introduces some challenges.
        Pointers and GEPs need to be converted to MemRefs and SubViews. Ensuring
        amdgpu completeness is another one.
    *   (Cost) XLA/GPU heavily relies on LLVM metadata:
        *   `range` for block/thread indices.
        *   `align`, `dereferenceable`, `invariant.load`, `alias.scope`,
            `noalias` for load/stores.
        *   `llvm.loop.unroll.disable`, `llvm.loop.unroll.full`,
            `llvm.loop.vectorize.enable` for sequential loops.
    *   (Benefit) Can be long-term. More portable.
1.  Refactor old emitters to be LHLO -> Linalg, and write new Linalg emitters
    *   (Cost) This is case by case. Compared to previous options, a new
        implementation that matches XLA's performance needs to go through the
        benchmark <-> optimize workflow, which can be a significant cost for
        some ops.
    *   (Benefit) unified stack; community support; portability; more
        optimization potentials.

## Prioritization

While all three tasks mentioned above are parallelizable, under limited
resources they have to be serialized. The prioritization focuses on visible
results for completion of each task.

The prioritization is: Task1 (LHLO for legacy emitters) > Task 2 (Thunks) > Task
3 (MLIR emitters).

By the end of Task 1, users of XLA can generate an LHLO (e.g. kernel generator)
and execute them. The compilation format will not be serializable MLIR.

By the end of Task 2, LHLO lowers to proper, serializable MLIR. This enables
offline compilation.

By the end of Task 3, all XLA emitters are MLIR-based in its implementation.

## Detailed Design

### Step 1: (Task 1) Complete LHLO and Make Legacy Emitters Take LHLO

This step makes all existing XLA/GPU emitters interact with MLIR ops. This step
is pure refactoring and NFC.

This step is mostly mechanical, but it's worth noticing the following
discrepancies between an unnested HloComputation and LHLO:

*   Each HloInstruction has direct access to its operands (a data-flow DAG). On
    contrary, each LHLO op only has access to its operand buffers (a bipartite
    between ops and buffers). LHLO ops have to go through use-def chains to
    access their operand ops.
*   Unnested legacy emitters empirically almost never access their operands. The
    only exception is kReduce.
*   Unnested legacy emitters access BufferAssignment only for getting slices,
    not for accessing aux data structures like dataflow\_analysis() or
    alias\_analysis(). llvm\_ir builds its own alias\_analysis() based on slice
    information.

The conclusion is that LHLO should fit right-in without major hassle.

### Step 2: (Optional) Profiling Support

**This step is only needed if we start to discard some of the XLA Thunk logic
(see the next step).**

Before actually turning on any MLIR-based emitters, we need profiling for
MLIR-based emitters.

Currently XLA performs its own profiling by calling into StreamExecutor's timer.
The timer under the hood inserts two events before and after a kernel launch,
and measures the sync time between these two events.

There are roughly three approaches to support profiling in MLIR:

*   Run a profiler end-to-end
*   Add a profile op for each op in LHLO, using an injected profiler.

The "end-to-end" approach is transparent to MLIR, but suffers the same problem
that makes XLA not use it in the first place: library calls collected by a
profiler (nvprof/...) can't easily relate to HLO ops. For example, cuDNN
launches multiple kernels for each HLO, and it's hard to tell which kernels
correspond to which HLO.

The "injected profiler" approach requires:

*   LHLO to take a profiler as a parameter.
*   inserting profile.start / profile.end before and after each op.
*   a pass from that lowers profile.{start,end} to a C++ implementation.

The exact profiling can't be easily done for MLIR-generated ops, since:

*   MLIR doesn't have a timer, nor it depends on TFRT / StreamExecutor.
*   MLIR doesn't easily call into C functions with complicated parameters.

### Step 3: (Task 2) Migrating Thunks

This step migrates all host ops and library calls. This step will eliminate most
of the thunks and produce serializable MLIR instead.

There are roughly three kinds of thunks:

*   KernelThunk, which launches a kernel.
*   Control flow thunks, which has host control flow logic (conditional, while,
    for, sequence) and launch body kernels.
*   Library thunks: cuDNN, cuBLAS, cuFFT, NCCL, etc.

The **bottom line** is to:

*   Create a Thunk dialect that provides (de)serialize logic for all existing
    C++-based Thunks.
*   Change emitters to emit a graph of Thunk dialect.

**Optionally**, we can relieve some thunks from C++ implementation. KernelThunk
can lower to the GPU LaunchKernelOp. Control flow thunks can leverage the CFG
Dialect for loops and conditions, combined with LaunchKernelOp. This optional
step requires profiling and stream support.

### Step 4: (Task 3) Migrated ElementalIrEmitter

Once profiling is ready, we can complete and tune all ElementalIrEmitter-based
emitters in MLIR. Then we turn them on by default, assuming that all of these
MLIR-based emitters use a single stream.

Notice that it's beneficial to migrate XLA/CPU's ElementalIrEmitter as well,
since they share a large portion of the code.

With all benchmarking and performance hunting done (TODO: define performance
parity), we turn on the new MLIR-based elemental emitter, and delete the legacy
ElementalIrEmitter.

This step also provides easy fusion transitions (nested ops) for the later
migration.

### Step 5: Multi-Stream Support or Drop

We can't delete
[some of the emitters](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/stream_assignment.cc#L140)
until we support it in MLIR, or we drop the feature. It's a relatively large
amount of work in MLIR and a small amount of gain for XLA. We should investigate
current users of multi-stream XLA/GPU users, and try to delete this feature if
reasonable.

### Step 6: (Task 3) Migrated Device Ops

This step migrates all unnested ops, then we can delete all unnested emitters.

This calls on a rewrite/refactor for kCopy and kReduce. kReduce is already
worked on for plenty, so the actual amount of work that needs to be done remains
to be seen.
