# ruy is not BLAS

ruy is a matrix multiplication library. Its focus is to cover the matrix
multiplication needs of TensorFlow Lite.

ruy supports both floating-point (like Eigen) and quantized (like gemmlowp).

## Status

ruy is very new, immature code. It has quite good test coverage, but the code is
in flux, lacks comments, needs more cleanup, and there are no design docs at the
moment.

We hope to improve on all that and integrate ruy into TensorFlow Lite, at first
as a non-default path for ARM A64 only, over the next few weeks [April 2019].

## Efficiency

ruy is designed to achieve maximal performance not just on very large sizes, as
is the focus of many established libraries, but on whatever are the actual sizes
and shapes of matrices most critical in current TensorFlow Lite applications.
This often means quite small sizes, e.g. 100x100 or even 50x50, and all sorts of
rectangular shapes.

ruy is currently only optimized for ARM A64; other architectures have only slow
reference code at the moment.

ruy is currently optimized only for the following combination of storage orders:
LHS = row-major, RHS = column-major, destination = column-major. All other
combinations of storage orders fall back to slow reference code at the moment.
