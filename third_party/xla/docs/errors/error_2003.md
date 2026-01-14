# Error code: E2003

**Category:** Compile Time: Mosaic Unproven Memory Access Alignment

This error occurs when the compiler analyzes a memory access operation (such as
`vector.load`, `vector.store`, `tpu.load`, or `tpu.store`) and cannot
statically prove that the dynamic index used for a specific dimension is a
multiple of the required **tiling size**.

**Sample Error Messages:**

```
INTERNAL: Mosaic failed to compile TPU kernel: cannot statically prove that index in dimension 1 is a multiple of 128

at location: ...

The MLIR operation involved:
  %14372 = "vector.load"(%14371, %93, %14363) : (memref<4x256xf32, #tpu.memory_space<vmem>>, index, index) -> vector<1x32xf32>
```

**XLA Backends:** TPU

## Overview

When your kernel loads or stores a vector, the memory address
(calculated from the base pointer plus the dynamic index) must align with the
vector's **tiling size** on the hardware. For example, if a dimension is tiled
by 128 elements, the dynamic index used to access it must be `0`, `128`, `256`,
etc. Note that many operations (like vector loads and stores) have no such
requirements for static indices.

The compiler enforces this requirement using **static analysis**. It traces the
history of the index variable back through the arithmetic operations that
produced it (e.g., multiplications, additions). If the compiler cannot guarantee
(at compile time) that the resulting value will always be divisible by the
tiling size, it raises this error.

The compiler treats "proven misalignment" and "unknown alignment" identically.
So if you use an index that is mathematically guaranteed to be misaligned (e.g.,
`i * 128 + 32`), the compiler will raise the same error.

So this error can occur when
1. You use a runtime variable (dynamic index) to access memory.
2. The index calculation logic is too complex for the compiler to analyze.
3. The index is mathematically valid but lacks an explicit proof in the code.
4. Static analysis determines "proven misalignment".

## Debugging

To resolve this error you have the following options:

### 1. Assert Alignment Explicitly

If you know your index is valid but the compiler cannot prove it, use the
`tpu.assume_multiple` operation. This acts as a promise to the compiler that a
value is divisible by a specific factor.

### 2. Use Aligned Loads and Rotate

In scenarios where the misalignment is intentional,
instead of loading a small, unaligned vector segment:

*   Load a larger, fully aligned tile and then rotate the values by a dynamic
    amount to shift the desired data into position (since vector slices with
    dynamic start indices are not supported). or
*   Reshape or pad the tensor so that the data starts at index 0 and the stride
    between accesses matches the hardware alignment.
    *   Example: If you iterate over chunks of size 32 starting at offset 1,
        your offsets are 1, 33, 65... (unaligned).
    *   Fix: Re-pack the data into a new tensor where the first chunk is at 0
        and the dimension is padded to 128. Your offsets become 0, 128, 256...,
        which satisfies the alignment requirement.

These methods consume more memory but often simplify kernel logic and eliminate
the need for manual alignment assertions.
