# Experimental code for the new TF-Lite convertor, and MLIR dialects and utilities for TensorFlow Lite.

This directory contains:

1. Experimental code for the new TF-Lite convertor.
2. Code for the TF-lite dialect [MLIR](https://github.com/tensorflow/mlir).

## API:

The API for converting TensorFlow models to TensorFlow Lite will be through
`tf.lite.TFLiteConverter`. All the conversion code is open sourced, and
the API will be integrated soon.

### The conversion process from TensorFlow to TensorFlow Lite includes the following major passes:

- Import from GraphDef, in .pb or .pbtxt  format, into MLIR.
- Raise to Control-flow-graph. Converts TF Control Flow dialect to TF dialect.
- The Canonicalization pass iteratively applies canonicalization
transformations in a greedy way until no further changes occur.
Canonicalization includes constant folding.
- The Legalize pass converts TensorFlow operations to TensorFlow Lite
ones. The operations that cannot be mapped to TensorFlow Lite dialect
are left as TensorFlow operations. Unsupported op handling follows the
proposed TFLite mechanism.
- Optimizations are performed in both the TF & TFLite dialect; aiming for small
size and high performance (among the core value proposition of
TensorFlow Lite models).
- The Export pass writes out TensorFlow Lite FlatBuffer format. This pass
operates on MLIR TensorFlow Lite dialect and is simple/direct translation.

