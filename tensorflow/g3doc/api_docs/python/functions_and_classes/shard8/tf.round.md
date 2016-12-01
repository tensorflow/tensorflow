### `tf.round(x, name=None)` {#round}

Rounds the values of a tensor to the nearest integer, element-wise.

Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use tf::cint.
For example:

```python
# 'a' is [0.9, 2.5, 2.3, 1.5, -4.5]
tf.round(a) ==> [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float32` or `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of same shape and type as `x`.

