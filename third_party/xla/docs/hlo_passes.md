# HLO Passes

This document outlines the [HLO](https://openxla.org/xla/terminology)
optimizations and transformations passes in the
[XLA compiler](https://openxla.org/xla/architecture).

## Introduction

A single HLO Pass can be comprised of one or many compiler optimizations and
transformations, and XLA provides several hundred such passes. HLO focuses only
on the shape (e.g. a 3x4 matrix) and the
[operation semantics](https://openxla.org/xla/operation_semantics) of the arrays
to make the optimization or transformation easier.

For example:

*   [`AlgebraicSimplifier`:](https://github.com/openxla/xla/blob/c37fc6a383b870f43cef82280418fcefcc90b0f8/xla/hlo/transforms/simplifiers/algebraic_simplifier.h#L417)
    A pass that performs a number of mostly arithmetic simplifications and
    optimizations. Including:

    *   When dividing by a constant, an optimization is performed to transform
        the operation to multiplication by the inversion of the constant.

*   [`HloRematerialization`:](https://github.com/openxla/xla/tree/main/xla/hlo/transforms/simplifiers/hlo_rematerialization.h)
    A pass that recomputes selected expressions in the computation to reduce
    memory pressure caused by long live ranges of array-shaped values.

## Developer details

The base class for HLO passes can be found in
[`xla/hlo/pass/hlo_pass_interface.h`](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h).
HLO pass should not extend this class directly but instead should extend
[`HloModulePass`](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h#L142).

See also
[XLA HLO Pass Framework](https://github.com/openxla/xla/tree/main/xla/hlo/pass#readme).

### Tooling and Testing

XLA comes with multiple command line tools, including the hlo-opt tool. This
tool allows execution of an individual pass independent of the given platform
compilation stages. For more information see
[Tooling](https://openxla.org/xla/tools#hlo-opt_hlo_pass_development_and_debugging).

For information on writing unit tests for HLO Passes see
[Testing HLO Passes](https://openxla.org/xla/test_hlo_passes).

## Hardware-independent HLO Pass Examples

This section describes a few examples of passes shared across XLA backends. Some
passes may be specialized for specific backends, but the high-level
functionality is similar.

Shared passes or hardware-independent passes can be found in
[`xla/hlo/transforms`](https://github.com/openxla/xla/tree/main/xla/hlo/transforms).

### Rematerialization

See also
[`HloRematerialization`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_rematerialization.h).

Selectively recomputes expressions within the HLO graph to reduce memory usage.
Trades off higher compute for lower memory usage. Can reduce memory usage by
tens of percent and is required to run many large models.

### Algebraic Simplifier

See also
[`AlgebraicSimplifier`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.h).

A grab bag of simplifications, optimizations, and canonicalizations. Analogous
to
[LLVM’s `instcombine` pass](https://llvm.org/docs/Passes.html#instcombine-combine-redundant-instructions).

### Constant Folding

See also
[`HloConstantFolding`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_constant_folding.h).

Replaces expressions which can be evaluated at compile time with their constant
equivalent.

### Dead Code Elimination

See also
[`HloDCE`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_dce.h)
.

Removes operations with unused results (fast implementation).

### Call Graph Flattening

See also
[`FlattenCallGraph`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/flatten_call_graph.h).

A legalization pass which converts the HLO call graph into a tree by cloning
computations. Required because memory is statically assigned to HLO operations
and not based on dynamic call context.

### Reshape Mover

See also
[`ReshapeMover`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/reshape_mover.h).

Reshapes and transposes can be expensive, especially on TPU. This pass moves and
reshapes and transposes across elementwise operations enabling the operations to
be merged or eliminated.

### Zero-sized HLO Elimination

See also
[`ZeroSizedHloElimination`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h).

HLO supports arrays of zero size (one or more dimensions has a bound of zero).
This pass simplifies the graph by replacing zero-sized operations with
zero-sized constants.

## TPU-specific HLO Pass Examples

Passes specific to the TPU backend.

### Model parallelism

The partitioning of an XLA program across multiple cores is performed at the HLO
level and the TPU HLO pipeline includes a number of passes for supporting
multi-core execution.

#### Spatial partitioning

See also
[`ShardingPropagation`](https://github.com/openxla/xla/blob/main/xla/service/sharding_propagation.h).

Pass to support dividing operations across devices along non-batch dimensions.

### Handling of bfloat16

See also
[`BFloat16ConversionFolding`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/bfloat16_conversion_folding.h),
[`BFloat16MixedPrecisionRemoval`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/float_normalization.h),
and
[`BFloat16Propagation`](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/bfloat16_propagation.h).

TPUs support bfloat16 as a lower-precision, more compact floating-point
representation than 32-bit floats. Using bfloat16 reduces memory footprint and
memory bandwidth. The TPU HLO pipeline includes various passes for replacing
floats with bfloat16 into the program and propagating the precision through the
graph.

### Legalization passes

See also
[`GatherExpander`](https://github.com/openxla/xla/blob/main/xla/service/gather_expander.h),
and
[`BatchNormExpander`](https://github.com/openxla/xla/blob/main/xla/service/batchnorm_expander.h).

Passes which transform unsupported HLO into a form which the backend can emit or
for which the backend produces a more efficient lowering.

## GPU-specific HLO Pass Example

Passes specific to the GPU backend are found in
[`xla/service/gpu`](https://github.com/openxla/xla/tree/main/xla/service/gpu).
These passes can be identified as classes defined in `namespace gpu`.

### cuDNN Rewriter

See also
[`CudnnFusedConvRewriter`](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h)
and
[`CudnnNormRewriter`](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/cudnn_norm_rewriter.h).

Rewrites fused convolution and norm operations into their respective library
calls in cuDNN.

## CPU-specific HLO Pass Examples

Passes specific to the CPU backend are found in
[`xla/service/cpu`](https://github.com/openxla/xla/tree/main/xla/service/cpu).
These passes can be identified as classes defined in `namespace cpu`.

### Convolution Canonicalization

See also
[`ConvCanonicalization`](https://github.com/openxla/xla/blob/main/xla/service/cpu/conv_canonicalization.h).

Canonicalizes convolutions so that they can be lowered to a fast implementation
in Eigen.

### Operation Parallelization

See also
[`ParallelTaskAssigner`](https://github.com/openxla/xla/blob/main/xla/service/cpu/parallel_task_assignment.h).

Partitions HLOs into tasks to run on separate threads.

## Analysis passes

Analysis passes are not considered "HLO passes" since they do not transform HLO
and may not extend `HloModulePass`. Shared analyses are found in
[`xla/hlo/analysis`](https://github.com/openxla/xla/tree/main/xla/hlo/analysis).

### Analysis Pass Examples

#### Dataflow Analysis

See also
[`HloDataflowAnalysis`](https://github.com/openxla/xla/tree/main/xla/hlo/analysis/hlo_dataflow_analysis.h).

Identifies all HLO values in the graph and their uses.

#### Alias Analysis

See also
[`HloAliasAnalysis`](https://github.com/openxla/xla/tree/main/xla/hlo/analysis/hlo_alias_analysis.h).

Identifies must-alias relationships between values in the program.

#### Computation Cost Analysis

See also
[`HloCostAnalysis`](https://github.com/openxla/xla/tree/main/xla/service/hlo_cost_analysis.h).

Computes FLOP count and memory usage for all operations in the program.

#### HLO Verification

See also
[`HloVerifier`](https://github.com/openxla/xla/tree/main/xla/service/hlo_verifier.h).

Verifies various invariants of the HLO graph.
