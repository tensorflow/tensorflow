A decorator for registering the shape function for an op type.

This decorator is only used when defining a new op type. A shape
function is a function from an `Operation` object to a list of
`TensorShape` objects, with one `TensorShape` for each output of the
operation.

For example, assuming that operations of type `"Sub"` take two
inputs `x` and `y`, and return a single output `x - y`, all with the
same shape, the following shape function would be registered:

```python
@tf.RegisterShape("Sub")
def _sub_shape(op):
  return [op.inputs[0].get_shape().merge_with(op.inputs[1].get_shape())]
```

The decorator argument `op_type` is the string type of an
operation. This corresponds to the `OpDef.name` field for the proto
that defines the operation.
- - -

#### `tf.RegisterShape.__init__(op_type)` {#RegisterShape.__init__}

Saves the `op_type` as the `Operation` type.


