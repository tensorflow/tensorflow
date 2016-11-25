### `tf.nn.fused_batch_norm(x, scale, offset, mean=None, variance=None, epsilon=0.001, data_format='NHWC', is_training=True, name=None)` {#fused_batch_norm}

Batch normalization.

As described in http://arxiv.org/abs/1502.03167.

##### Args:


*  <b>`x`</b>: Input `Tensor` of 4 dimensions.
*  <b>`scale`</b>: A `Tensor` of 1 dimension for scaling.
*  <b>`offset`</b>: A `Tensor` of 1 dimension for bias.
*  <b>`mean`</b>: A `Tensor` of 1 dimension for population mean used for inference.
*  <b>`variance`</b>: A `Tensor` of 1 dimension for population variance
            used for inference.
*  <b>`epsilon`</b>: A small float number added to the variance of x.
*  <b>`data_format`</b>: The data format for x. Either "NHWC" (default) or "NCHW".
*  <b>`is_training`</b>: A bool value to specify if the operation is used for
               training or inference.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`y`</b>: A 4D Tensor for the normalized, scaled, offsetted x.
*  <b>`batch_mean`</b>: A 1D Tensor for the mean of x.
*  <b>`batch_var`</b>: A 1D Tensor for the variance of x.

##### Raises:


*  <b>`ValueError`</b>: If mean or variance is not None when is_training is True.

