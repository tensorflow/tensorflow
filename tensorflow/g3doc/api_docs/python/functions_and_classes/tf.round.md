### `tf.round(x, name=None)` {#round}

Rounds the values of a tensor to the nearest integer, element-wise.

For example:

```python
# 'a' is [0.9, 2.5, 2.3, -4.4]
tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float` or `double`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of same shape and type as `x`.

