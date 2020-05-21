# Convolution Emitter

## Context

This is a doc that describes a set of patches that are still under review.
TODO(timshen): Change once all patches are checked in.

The convolution emitter is a prototype with the following goals:

*   The top priority is performance.
*   It supports arbitrarily sophiscated layouts.
*   It supports platform-specific high-performance instructions.
*   It is as portable as possible.
*   It enables fusion support in the future.

## Current Design

### Overview

The prototype consists of the following components:

*   The emitter currently focuses on NVIDIA Volta architecture and N(C/4)HW4
    layout.
*   An MLIR-based emitter. It takes a set of tuning parameters and a convolution
    configuration, then produces a NVVM device function.
*   An autotuner, which generates tuning parameters given a convolution
    configuration.
*   A test framework, which executes the generated device function with random
    inputs, and compares the result against cuDNN.

### The Emitter - Naive Implementation

The emitter starts with a hand-built, naive implementation that looks like
following Resnet first layer convolution (pseudo code):

```mlir
func @Conv(%input : memref<128x1x224x224xvector<4xf16>>,
           %filter : memref<64x1x7x7xvector<4xf16>>,
           %output : memref<128x64x224x224xf16>) {
  affine.parallel (%n, %o, %oh, %ow) = 0 to 128, 0 to 64, 0 to 112, 0 to 112 {
    %acc = alloc() : memref<f32>
    affine.store 0, %acc[]
    affine.for (%c, %fh, %fw) = 0 to 1, 0 to 7, 0 to 7 {
      %a = affine.padded.load %input[%n, %c, %oh * 2 + %fh - 3, %ow * 2 + %fw - 3]
      %b = affine.load %filter[%o, %c, %fh, %fw]
      %c = affine.load %acc[]
      %d = std.fpext %a to vector<4xf32>
      %e = std.fpext %b to vector<4xf32>
      %f = std.multiply %d, %e
      %g = "reduce" %f
      %v = %g + %c
      affine.store %v, %acc[]
    }
    %c = affine.load %acc[]
    affine.store %acc, %output[%n, %o, %oh, %ow]
  }
}
```

A few extensions are used in the example above:

*   affine.padded.load allows out-of-bounds access, in which case the result is
    always 0.
*   The "reduce" operation produces the sum of elements in a vector.

Also notice that the input element type is vector<4xf16> only because the
current implementation does so. A MemRef with <...x4xf16> should work as well,
given the alignment properly aligned to at least 8 (usually 16).

Then the emitter does a few semantic preserving transformations to work the code
towards PTX's structure.

### The Emitter - Tiling

The following is the naive code after loop tiling:

```mlir
func @Conv(%input : memref<128x1x224x224xvector<4xf16>>,
           %filter : memref<64x1x7x7xvector<4xf16>>,
           %output : memref<128x64x224x224xf16>) {
  affine.parallel (%n0, %o0, %oh0, %ow0) = 0 to 128, 0 to 1, 0 to 7, 0 to 7 {
    affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
      %acc = alloc() : memref<f32>
      affine.store 0, %acc[]
      affine.for (%c0, %fh0, %fw0) = 0 to 1, 0 to 1, 0 to 1 {
        affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
          %a = affine.padded.load %input[
              %n0 * 1 + %n1,
              %c0 * 1 + %c1,
              (%oh0 * 16 + %oh1) * 2 + %fh0 * 7 + %fh1 - 3,
              (%ow0 * 16 + %ow1) * 2 + %fw0 * 7 + %fw1 - 3]
          %b = affine.load %filter[
              %o0 * 64 + %o1,
              %c0 * 1 + %c1,
              %fh0 * 7 + %fh1,
              %fw0 * 7 + %fw1]
          %old = affine.load %acc[]
          %d = std.fpext %a to vector<4xf32>
          %e = std.fpext %b to vector<4xf32>
          %f = std.multiply %d, %e
          %g = "reduce" %f
          %new = %g + %old
          affine.store %new, %acc[]
        }
      }
      %v = affine.load %acc[]
      affine.store %v, %output[
          %n0 * 1 + %n1,
          %o0 * 64 + %o1,
          %oh0 * 16 + %oh1,
          %ow0 * 16 + %ow1]
    } { ptx_block }
  } { ptx_grid }
}
```

The motivation is obvious - we need to decide which loops are parallelized on
the compute units in the PTX architecture. The `ptx_grid` and `ptx_block`
directs that the loop should be parallelized on a grid / a block, respectively.

Also notice that to keep the code pattern clean and neat, tiling is implemented
in the following way. Defining "simple loop" as a loop with lower bound 0, and
step 1, the tiling:

*   only takes simple loops.
*   only produces simple loops.
*   no extra operation is generated. All altered index calculations are done in
    each user AffineMaps.

The contracting dimensions (%c, %fh, %fw) are also tiled for once. The
significance will be seen later in shared memory promotion.

### The Emitter - Splitting

This step splits the body of the (%n1, %o1, %oh1, %ow1) loop into several parts:

*   The code that sets the accumulators to 0.
*   The actual convolution computation code.
*   The code that writes back accumulators to the %output buffer.

This transformation "vectorizes" the accumulator accordingly as the `alloc()`
gets hoisted out of the `affine.parallel` op.

After splitting:

```mlir
func @Conv(%input : memref<128x1x224x224xvector<4xf16>>,
           %filter : memref<64x1x7x7xvector<4xf16>>,
           %output : memref<128x64x224x224xf16>) {
  affine.parallel (%n0, %o0, %oh0, %ow0) = 0 to 128, 0 to 1, 0 to 7, 0 to 7 {
    %acc = alloc() : memref<1x64x16x16xf32>
    affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
      affine.store 0, %acc[%n1, %o1, %oh1, %ow1]
    } { ptx_block }
    affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
      affine.for (%c0, %fh0, %fw0) = 0 to 1, 0 to 1, 0 to 1 {
        affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
          %a = affine.padded.load %input[
              %n0 * 1 + %n1,
              %c0 * 1 + %c1,
              (%oh0 * 16 + %oh1) * 2 + %fh0 * 7 + %fh1 - 3,
              (%ow0 * 16 + %ow1) * 2 + %fw0 * 7 + %fw1 - 3]
          %b = affine.load %filter[
              %o0 * 64 + %o1,
              %c0 * 1 + %c1,
              %fh0 * 7 + %fh1,
              %fw0 * 7 + %fw1]
          %old = affine.load %acc[%n1, %o1, %oh1, %ow1]
          %d = std.fpext %a to vector<4xf32>
          %e = std.fpext %b to vector<4xf32>
          %f = std.multiply %d, %e
          %g = "reduce" %f
          %new = %g + %old
          affine.store %new, %acc[%n1, %o1, %oh1, %ow1]
        }
      }
    } { ptx_block }
    affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
      %v = affine.load %acc[%n1, %o1, %oh1, %ow1]
      affine.store %v, %output[
          %n0 * 1 + %n1,
          %o0 * 64 + %o1,
          %oh0 * 16 + %oh1,
          %ow0 * 16 + %ow1]
    } { ptx_block }
  } { ptx_grid }
}
```

To prepare for the next transformations, we'd also like to sink the (%n1, %o1,
%oh1, %ow1), as (%c0, %fh0, %fw0) is not interesting.

```
affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
  affine.for (%c0, %fh0, %fw0) = 0 to 1, 0 to 1, 0 to 1 {
    affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
      ...
    }
  }
} { ptx_block }

=>

affine.for (%c0, %fh0, %fw0) = 0 to 1, 0 to 1, 0 to 1 {
  affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
    affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
      ...
    } { ptx_block }
  }
}
```

### The Emitter - Shared Memory Promotion

This transformation is done by `affineDataCopyGenerate`, which does precise
calculation on how much memory is transferred for a load operation.

After calculating the sizes of the shared memory buffer (`%promoted_input` and
`%promoted_filter`), the transformation also creates loads and stores to
pre-fetch data from global memory (`%input`, `%filter`) to the promoted, shared
memory.

```mlir
// Before
affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
  affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
    %a = affine.padded.load %input[
        %n0 * 1 + %n1,
        %c0 * 1 + %c1,
        (%oh0 * 16 + %oh1) * 2 + %fh0 * 7 + %fh1 - 3,
        (%ow0 * 16 + %ow1) * 2 + %fw0 * 7 + %fw1 - 3]
    %b = affine.load %filter[
        %o0 * 64 + %o1,
        %c0 * 1 + %c1,
        %fh0 * 7 + %fh1,
        %fw0 * 7 + %fw1]
    %old = affine.load %acc[%n1, %o1, %oh1, %ow1]
    %d = std.fpext %a to vector<4xf32>
    %e = std.fpext %b to vector<4xf32>
    %f = std.multiply %d, %e
    %g = "reduce" %f
    %new = %g + %old
    affine.store %new, %acc[%n1, %o1, %oh1, %ow1]
  } { ptx_block }
}
```

```mlir
// After

%promoted_input = alloc() : memref<1x1x37x37, memory_space = 3>
%promoted_filter = alloc() : memref<64x1x7x7, memory_space = 3>
affine.parallel (%i0, %i1, %i2, %i3) = 0 to 1, 0 to 1, 0 to 37, 0 to 37 {
  %v = affine.padded.load %input[
      %n0 * 1 + %i0,
      %c0 * 1 + %i1,
      (%oh0 * 16) * 2 + %fh0 * 7 + %i2 - 3,
      (%ow0 * 16) * 2 + %fw0 * 7 + %i3 - 3]
  affine.store %v, %promoted_input[%i0, %i1, %i2, %i3]
} { ptx_block }
affine.parallel (%i0, %i1, %i2, %i3) = 0 to 64, 0 to 1, 0 to 7, 0 to 7 {
  %v = affine.load %filter[
      %o0 * 64 + %i0,
      %c0 * 1 + %i1,
      %fh0 * 7 + %i2,
      %fw0 * 7 + %i3]
  affine.store %v, %promoted_filter[%i0, %i1, %i2, %i3]
} { ptx_block }
affine.for (%c1, %fh1, %fw1) = 0 to 1, 0 to 7, 0 to 7 {
  affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
    %a = affine.load %promoted_input[%n1, %c1, %oh1 * 2 + %fh1, %ow1 * 2 + %fw1]
    %b = affine.load %promoted_filter[%o1, %c1, %fh1, %fw1]
    %old = affine.load %acc[%n1, %o1, %oh1, %ow1]
    %d = std.fpext %a to vector<4xf32>
    %e = std.fpext %b to vector<4xf32>
    %f = std.multiply %d, %e
    %g = "reduce" %f
    %new = %g + %old
    affine.store %new, %acc[%n1, %o1, %oh1, %ow1]
  } { ptx_block }
}
```

### The Emitter - Volta MMA Instruction

This transformation turns the inner loop:

```mlir
affine.parallel (%n1, %o1, %oh1, %ow1) = 0 to 1, 0 to 64, 0 to 16, 0 to 16 {
  %a = affine.load %promoted_input[%n1, %c1, %oh1 * 2 + %fh1, %ow1 * 2 + %fw1]
  %b = affine.load %promoted_filter[%o1, %c1, %fh1, %fw1]
  %old = affine.load %acc[%n1, %o1, %oh1, %ow1]
  %d = std.fpext %a to vector<4xf32>
  %e = std.fpext %b to vector<4xf32>
  %f = std.multiply %d, %e
  %g = "reduce" %f
  %new = %g + %old
  affine.store %new, %acc[%n1, %o1, %oh1, %ow1]
} { ptx_block }
```

to multiple Volta mma.sync instructions. The result is not shown here, because
the prototype currently only hacks it up to achieve benchmark goals.

### The Autotuner

As shown above, many parameters dictate how a naive implementation is
transformed. For now, the parameters are all tile sizes. On the top of the
emitter, the prototype includes a simple autotuner that enumerates all good
combinations of tile sizes and invoke the emitter with each of the combinations.
With the assistance of in-process benchmarking, the autotuner is able to pick
the best set of parameters.

## Future Improvements

*   Explore Linalg/Vector for a higher-level naive implementation. MMA
    instruction handling would be much easier with high-level functional
    constructs.
*   Explore other layouts. The current layout corresponds to NVIDIA
    `CUDNN_TENSOR_NCHW_VECT_C` but for fp16s.
*   Iron out GPU dialect related lowering. Annotations like `ptx_grid` and
    `ptx_block` should be generalized to more architectures.
*   Speed up autotuning through more pruning.
*   Support dynamic shapes.
