# Known Issues

Compilation with XLA can greatly improve the performance of your programs, but
the TensorFlow interop has a number of known sharp corners.

## TensorArray TF/XLA interconversion is not supported

*Error message*: `Support for TensorList crossing the XLA/TF boundary is not
implemented`.

XLA supports `tf.TensorArray`. However, the _interconversion_ between TF and XLA
representations is not implemented yet. This error often arises when the
`TensorArray` is used inside the compiled block, but the derivative is taken
outside.

*Workaround*: compile the outermost scope which is taking the derivative.

## TensorFlow while loops need to be bounded (or have backprop disabled)

*Error message*: `XLA compilation requires a fixed tensor list size. Set the max
number of elements. This could also happen if you're using a TensorArray in a
while loop that does not have its maximum_iteration set, you can fix this by
setting maximum_iteration to a suitable value`.

TF while [loops](https://www.tensorflow.org/api_docs/python/tf/while_loop)
created using `tf.while_loop` support backpropagation by accumulating all
intermediate results in a `TensorArray`, but XLA only supports bounded
`TensorArray`s.

*Workaround*: all compiled while loops need to either have `maximum_iterations`
parameter set to a constant value known at compile time, or backpropagation
disabled using `back_prop=False`.

## Dynamic `tf.TensorArray` is not supported

Writes into `tf.TensorArray(..., dynamic_size=True)` are not compilable with
XLA, as such writes require an unknown number of reallocations when the array
exceeds the original bound.

*Workaround*: provide a statically known bound to your arrays.

## Random number generation ignores TF seed

XLA currently ignores TF seeds to random operations. This affects stateful TF
random operations, such as `tf.random.normal`, or `tf.nn.dropout`. XLA will
behave as if the compilation was seeded with a new unique seed at each run. This
limitation does not apply to stateless random ops.

## TensorFlow Asserts are ignored

Assertions created using `tf.Assert` and similar functions are noops when
compiled to XLA. While proper assertion support is in principle possible, it
might make certain optimizations impossible (mainly fusing the buffer on which
the assertion is performed).

## Non-deterministic output

On CPU and GPU the output may be non-deterministic (same as TF proper).

*Workaround*: To enforce determinism, set the `TF_DETERMINISTIC_OPS` environment
variable to `1` (same as for TF).
