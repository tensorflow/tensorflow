# Error code: E2002

**Category:** Compile Time: Mosaic Input/Output Misaligned Block and Tiling

This error occurs when the block shape of a kernel input or output does not
align with the default tiling of the datatype on the specific TPU hardware
being used.

**Sample Error Messages:**

```
UNIMPLEMENTED: Mosaic failed to compile TPU kernel: Failed to set window params
for input 0: Operand of shape (..., 256, 8192) has tiling (16, 128), but its
block shape (..., 8, 8192) is not divisible by tiling evenly nor matches the
full shape.
```

**XLA Backends:** TPU

## Overview

Tensor cores (TC) in TPUs have two-dimensional vector registers. The two
dimensions are called **sublane** and **lane**. Since TPU compute units (e.g.,
MXU) operate at the granularity of vector registers, XLA arrays are laid out in
TPU memory in tiles.

This tiling (e.g., `8x128`) minimizes data transformations when feeding the
compute units. The exact tiling dimensions (sublanes × lanes) depend on the
hardware generation and the data type. For example, a common tiling for
most types is **8×128**.

At compile time, XLA enforces the following constraints for the minor and
2nd-minor dimensions of each kernel input/output:

1.  **Divisibility:** The block dimension must be a multiple of the tile
    dimension in the underlying tensor, or
2.  **Full Shape Exception:** If the block dimension is not divisible, it must
    be equal to the **full size** of that dimension in the underlying tensor.

This error is triggered when a block violates both conditions. For example,
loading a block of shape `(8, 100)` from a input of shape `(8, 1024)` on
hardware with shape `(8, 128)` tiling fails because `100` is not divisible by
`128` and `100 != 1024`. But, it would be allowed if the input shape was
`(32, 100)`.

## Debugging

To resolve this error, ensure your kernel's block shapes align with the
current hardware tiling. Modify your kernel code to align the block size such
that it is a multiple of the required tiling.

* **Example:** If the error states the tiling is `(16, 128)` but your block
shape is `(8, 128)`, change the block spec such that the shape matches
`(16, 128)`.
