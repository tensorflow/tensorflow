# AutoGraph reference

[Index](index.md)

## Common AutoGraph errors

### "WARNING: AutoGraph could not transform `<name>`"

This warning is output when AutoGraph could not convert a function, for an
unexpected reason. The error message contains the reason why the function could
not be converted, as well as guidance on how to proceed next.

The exact error message may vary from version to version but in general, the
cause of the failure appears somewhere in the text, for example as
"Cause: could not get source code" or "Original error: could not get source
code".

Note: AutoGraph does not always output a warning. For example, constructors
are silently called without conversion.

When this warning is printed, the code returned by AutoGraph still executes, but
the functions indicated in the warning will be executed as they are, without
conversion. If the functions contain pure Python or graph code (for example,
they have no Tensor-dependent control flow), then the code is likely to still
run without error. However, if it contains any constructs that are only
supported in AutoGraph, expect subsequent exceptions.

Note: the warning is output to the [abseil](https://github.com/abseil/abseil-py)
logger, with `WARNING` severity. To direct these warnings to `stdout`, use
`tf.autograph.set_verbosity(0, True)`.

### "Unable to locate the source code" or "Source not found" errors

Newer versions of AutoGraph raise a `ConversionError`. Older versions print a
warning. In both cases, a similar message about finding the source code is
included.

These errors are raised when AutoGraph is unable to find the source code of
functions it needs to transform. See [Limitations](limitations.md) for more
details.

### "WARNING: Large unrolled loop detected"

This warning is output when AutoGraph detects a `for` or `while` loop that
creates TensorFlow ops and which has a large number of iterations and creates.

This usually indicates a loop that was intended to run as a `tf.while_loop`, but
instead runs as a Python loop.

For example, a training loop might mistakenly iterate over a Python `range`,
instead of `tf.range`:

```
num_steps = 10000
step = tf.constant(0)
for i in range(num_steps):
  step += 1
  train_step(model)
```

Another example is when using custom generators which AutoGraph does not
support, even if they wrap over supported iterators like Datasets:

```
def my_iterator(ds):
  for data in ds:
    yield data

# Custom iterators always dispatch to a Python for loop.
for x in my_iterator(tf.data.Dataset.range(10)):
  tf.print(x)
```

Note: This verification is only performed when `__debug__` is `True`.

Note: the warning is output to the [abseil](https://github.com/abseil/abseil-py)
logger, with `WARNING` severity. To direct these warnings to `stdout`, use
`tf.autograph.set_verbosity(0, True)`.

### "OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool`"

This exception is raised whenever a `tf.Tensor` is type-cast as a Python `bool`,
in a context where eager execution is not active. The exception is only raised
when graph execution is active, for example inside a `@tf.function` with
AutoGraph turned off.

**When AutoGraph is on**, it can be caused by:
  * placing a Tensor-dependent `break`, `continue` or `return` inside a Python
    loop (see example below)
  * attempting to use a `tf.Tensor` in a list comprehension, by iterating over
    it or using it in a condition)

A typical example of mixing Python and TF control flow in an incompatible way
is:

```
for i in range(3):  # Python loop
  if i > tf.constant(0):  # TF conditional
    break  # raises OperatorNotAllowedInGraphError
```

The way these errors are typically fixed is by ensuring all control flow is
TF control flow:

```
for i in tf.range(3):  # TF loop
  if i > tf.constant(0):  # TF conditional
    break  # works
```

**When AutoGraph is off**, it can be caused by using a `tf.Tensor` value as:

  * the condition of an `if` or `while` statement: `if <tensor>:`
  * the argument in a logical expression: `tensor and another_tensor`
  * the argument to the `bool` built-in: `bool(tensor)`

Note: These operations are allowed when executing eagerly.

When encountering this error, make sure that the function is either decorated
with `@tf.function`, or called from another function decorated in this way. Also
look at the console and logging output for conversion warnings (see the section
above).

### "OperatorNotAllowedInGraphError: iterating over `tf.Tensor`"

This exception is raised whenever you try to iterate over a `tf.Tensor`,
in a context where eager execution is not active. The exception is only raised
when graph execution is active, for example inside a `@tf.function` with
AutoGraph turned off. It can be caused by using a `tf.Tensor` value as:

  * the iterated of a `for` statement: `for i in tensor:`
  * the argument to the `iter` built-in: `iter(tensor)`

Note: These operations are allowed when executing eagerly.

This exception is similar to the previous example, and has similar causes and
remedies.

### "InaccessibleTensorError: The tensor `<name>` is defined in another function or code block"

This exception is common to code which attempts to obtain values calculated
within a `tf.cond`, `tf.while_loop`, or another `@tf.function` without using
functional style or through mutable collections. See
[Limitations](limitations.md) for more details.

### "StagingError: in converted code"

This exception is used by AutoGraph to wrap exceptions with custom constructors
that it cannot re-raise with the original type. See
[Error handling](error_handling.md) for more details. If your code uses custom
exceptions, expect them to be wrapped by this exception.

### "Unable to identify source code of lambda function"

This error usually appears in the context of a conversion warning. It indicates
that a lambda function could not be parsed (see [Limitations](limitations.md)).

This type of errors can usually be avoided by creating lambda functions in
separate simple assignments, for example:

```
l = lambda <args>: <body>
```
