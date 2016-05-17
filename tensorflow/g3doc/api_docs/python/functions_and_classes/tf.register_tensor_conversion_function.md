### `tf.register_tensor_conversion_function(base_type, conversion_func, priority=100)` {#register_tensor_conversion_function}

Registers a function for converting objects of `base_type` to `Tensor`.

The conversion function must have the following signature:

    def conversion_func(value, dtype=None, name=None, as_ref=False):
      # ...

It must return a `Tensor` with the given `dtype` if specified. If the
conversion function creates a new `Tensor`, it should use the given
`name` if specified. All exceptions will be propagated to the caller.

The conversion function may return `NotImplemented` for some
inputs. In this case, the conversion process will continue to try
subsequent conversion functions.

If `as_ref` is true, the function must return a `Tensor` reference,
such as a `Variable`.

NOTE: The conversion functions will execute in order of priority,
followed by order of registration. To ensure that a conversion function
`F` runs before another conversion function `G`, ensure that `F` is
registered with a smaller priority than `G`.

##### Args:


*  <b>`base_type`</b>: The base type or tuple of base types for all objects that
    `conversion_func` accepts.
*  <b>`conversion_func`</b>: A function that converts instances of `base_type` to
    `Tensor`.
*  <b>`priority`</b>: Optional integer that indicates the priority for applying this
    conversion function. Conversion functions with smaller priority values
    run earlier than conversion functions with larger priority values.
    Defaults to 100.

##### Raises:


*  <b>`TypeError`</b>: If the arguments do not have the appropriate type.

