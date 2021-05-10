# TensorFlow Lite and TensorFlow operator compatibility

TensorFlow Lite supports a number of TensorFlow operations used in common
inference models. As they are processed by the TensorFlow Lite Optimizing
Converter, those operations may be elided or fused, before the supported
operations are mapped to their TensorFlow Lite counterparts.

Since the TensorFlow Lite builtin operator library only supports a limited
number of TensorFlow operators, not every model is convertible. Even for
supported operations, very specific usage patterns are sometimes expected, for
performance reasons. We expect to expand the set of supported operations in
future TensorFlow Lite releases.

The best way to understand how to build a TensorFlow model that can be used with
TensorFlow Lite is to carefully consider how operations are converted and
optimized, along with the limitations imposed by this process.

## Supported types

Most TensorFlow Lite operations target both floating-point (`float32`) and
quantized (`uint8`, `int8`) inference, but many ops do not yet for other types
like `tf.float16` and strings.

Apart from using different version of the operations, the other difference
between floating-point and quantized models is the way they are converted.
Quantized conversion requires dynamic range information for tensors. This
requires "fake-quantization" during model training, getting range information
via a calibration data set, or doing "on-the-fly" range estimation. See
[quantization](../performance/model_optimization.md).

## Supported operations and restrictions

TensorFlow Lite supports a subset of TensorFlow operations with some
limitations. For full list of operations and limitations see
[TF Lite Ops page](https://www.tensorflow.org/mlir/tfl_ops).

## Straight-forward conversions, constant-folding and fusing

A number of TensorFlow operations can be processed by TensorFlow Lite even
though they have no direct equivalent. This is the case for operations that can
be simply removed from the graph (`tf.identity`), replaced by tensors
(`tf.placeholder`), or fused into more complex operations (`tf.nn.bias_add`).
Even some supported operations may sometimes be removed through one of these
processes.

Here is a non-exhaustive list of TensorFlow operations that are usually removed
from the graph:

*   `tf.add`
*   `tf.check_numerics`
*   `tf.constant`
*   `tf.div`
*   `tf.divide`
*   `tf.fake_quant_with_min_max_args`
*   `tf.fake_quant_with_min_max_vars`
*   `tf.identity`
*   `tf.maximum`
*   `tf.minimum`
*   `tf.multiply`
*   `tf.no_op`
*   `tf.placeholder`
*   `tf.placeholder_with_default`
*   `tf.realdiv`
*   `tf.reduce_max`
*   `tf.reduce_min`
*   `tf.reduce_sum`
*   `tf.rsqrt`
*   `tf.shape`
*   `tf.sqrt`
*   `tf.square`
*   `tf.subtract`
*   `tf.tile`
*   `tf.nn.batch_norm_with_global_normalization`
*   `tf.nn.bias_add`
*   `tf.nn.fused_batch_norm`
*   `tf.nn.relu`
*   `tf.nn.relu6`

Note: Many of those operations don't have TensorFlow Lite equivalents, and the
corresponding model will not be convertible if they can't be elided or fused.

## Experimental Operations
The following TensorFlow Lite operations are present, but not ready for custom
models:

*   `CALL`
*   `CONCAT_EMBEDDINGS`
*   `CUSTOM`
*   `EMBEDDING_LOOKUP_SPARSE`
*   `HASHTABLE_LOOKUP`
*   `LSH_PROJECTION`
*   `SKIP_GRAM`
*   `SVDF`
