### `tf.nn.quantized_relu_x(features, max_value, min_features, max_features, out_type=None, name=None)` {#quantized_relu_x}

Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
*  <b>`max_value`</b>: A `Tensor` of type `float32`.
*  <b>`min_features`</b>: A `Tensor` of type `float32`.
    The float value that the lowest quantized value represents.
*  <b>`max_features`</b>: A `Tensor` of type `float32`.
    The float value that the highest quantized value represents.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (activations, min_activations, max_activations).

*  <b>`activations`</b>: A `Tensor` of type `out_type`. Has the same output shape as "features".
*  <b>`min_activations`</b>: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
*  <b>`max_activations`</b>: A `Tensor` of type `float32`. The float value that the highest quantized value represents.

