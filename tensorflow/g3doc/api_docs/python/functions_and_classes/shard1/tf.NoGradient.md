### `tf.NoGradient(op_type)` {#NoGradient}

Specifies that ops of type `op_type` do not have a defined gradient.

This function is only used when defining a new op type. It may be
used for ops such as `tf.size()` that are not differentiable.  For
example:

```python
tf.NoGradient("Size")
```

##### Args:


*  <b>`op_type`</b>: The string type of an operation. This corresponds to the
    `OpDef.name` field for the proto that defines the operation.

##### Raises:


*  <b>`TypeError`</b>: If `op_type` is not a string.

