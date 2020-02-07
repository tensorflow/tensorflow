# The ruy matrix multiplication library

This is not an officially supported Google product.

ruy is a matrix multiplication library. Its focus is to cover the matrix
multiplication needs of neural network inference engines. Its initial user has
been TensorFlow Lite, where it is used by default on the ARM CPU architecture.

ruy supports both floating-point and 8bit-integer-quantized matrices.

## Efficiency

ruy is designed to achieve maximal performance not just on very large sizes, as
is the focus of many established libraries, but on whatever are the actual sizes
and shapes of matrices most critical in current TensorFlow Lite applications.
This often means quite small sizes, e.g. 100x100 or even 50x50, and all sorts of
rectangular shapes.

ruy is currently only optimized for the ARM architectures (both 64-bit and
32-bit code). Optimization for the Intel x86 architecture is in progress.

ruy is currently optimized only for the following combination of storage orders:
LHS = row-major, RHS = column-major, destination = column-major. All other
combinations of storage orders fall back to slow reference code at the moment.
