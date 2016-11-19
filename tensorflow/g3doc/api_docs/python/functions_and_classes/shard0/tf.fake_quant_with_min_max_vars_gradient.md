### `tf.fake_quant_with_min_max_vars_gradient(gradients, inputs, min, max, name=None)` {#fake_quant_with_min_max_vars_gradient}

Compute gradients for a FakeQuantWithMinMaxVars operation.

##### Args:


*  <b>`gradients`</b>: A `Tensor` of type `float32`.
    Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
*  <b>`inputs`</b>: A `Tensor` of type `float32`.
    Values passed as inputs to the FakeQuantWithMinMaxVars operation.
    min, max: Quantization interval, scalar floats.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

*  <b>`backprops_wrt_input`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. inputs:
    `gradients * (inputs >= min && inputs <= max)`.
*  <b>`backprop_wrt_min`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. min parameter:
    `sum(gradients * (inputs < min))`.
*  <b>`backprop_wrt_max`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. max parameter:
    `sum(gradients * (inputs > max))`.

