### `tf.div(x, y, name=None)` {#div}

Divides x / y elementwise (using Python 2 division operator semantics).

NOTE: Prefer using the Tensor division operator or tf.divide which obey Python
division operator semantics.

This function divides `x` and `y`, forcing Python 2.7 semantics. That is,
if one of `x` or `y` is a float, then the result will be a float.
Otherwise, the output will be an integer type. Flooring semantics are used
for integer division.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` returns the quotient of x and y.

