# Known Issues

Compilation with XLA can greatly improve the performance of your programs, but
the TensorFlow interop has a number of known sharp corners.

## TensorArray TF/XLA interconversion

The problem manifests itself as an error message
`Support for TensorList crossing the XLA/TF boundary is not implemented`.

XLA supports `tf.TensorArray`. However, the _interconversion_ between TF and
XLA representations is not implemented yet.
This error often arises when the `TensorArray` is used inside the compiled
block, but the derivative is taken outside.

Workaround: compile the outermost scope which is taking the derivative.

## Dynamic `tf.TensorArray` is not supported

Writes into `tf.TensorArray(..., dynamic_size=True)` are not compilable with
XLA, as such writes require an unknown number of reallocations when the array
exceeds the original bound.

Workaround: provide a statically known bound to your arrays.

## Random number generation

XLA currently ignores TF seeds to random operations. This affects stateful TF
random operations, such as `tf.random.normal`, or `tf.nn.dropout`.  XLA will
behave as if the compilation was seeded with a new unique seed at each run. This
limitation does not apply to stateless random ops.

