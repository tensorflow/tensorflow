# Known Issues

Compilation with XLA can greatly improve the performance of your programs, but
the TensorFlow interop has a number of known sharp corners.

## `tf.Variable` on a different device

*Error message*: `INVALID_ARGUMENT: Trying to access resource <Variable>
(defined @ <Loc>) located in device CPU:0 from device GPU:0`

XLA cluster runs on exactly one device, and it can not read or write to
`tf.Variable` located on a different device. Usually this error message
indicates that the variable was not placed on the right device to begin with.
The error message should precisely specify the location of the offending
variable.

NOTE: `tf.Variable` of type `int32` are always placed on a host, and can not be
placed on a GPU. As a workaround, `int64` can be used.

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
behave as if the compilation was seeded with a new unique seed at each run
within the same process (the first run of the process will always yield the same
result).

*Workaround*: use
[the recommended RNGs](https://www.tensorflow.org/guide/random_numbers#stateless_rngs)
such as `tf.random.stateless_uniform` or the `tf.random.Generator` directly.

## Must-be-constant inputs which are functions of induction variables are not supported

*Error Message*: `XLA compilation requires that operator arguments that
represent shapes or dimensions be evaluated to concrete values at compile time.
This error means that a shape or dimension argument could not be evaluated at
compile time, usually because the value of the argument depends on a parameter
to the computation, on a variable, or on a stateful operation such as a random
number generator`.

XLA requires certain values to be known at compile time, such as reduction axis
of a reduce operation, or transposition dimensions. Consider the case when e.g.
reduction axis is defined as a function of an induction variable of `tf.range`:
resolving it statically is not possible without unrolling the entire loop, which
might not be desired by the user.

*Workaround*: Unroll loops, e.g. by converting `tf.range` into Python `range`.

NOTE: The error message above is not unique to this issue, and can arise due to
other limitations or bugs.
