# AutoGraph reference

[Index](index.md)

## Common AutoGraph errors

### "WARNING: AutoGraph could not transform `<name>`"

This warning is output when AutoGraph could not convert a function, for an
unexpected reason. The error message contains the reason why the function could
not be converted, as well as guidance on how to proceed next.

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

### "OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool`"

This exception is raised whenever a `tf.Tensor` is type-cast as a Python `bool`,
in a context where eager execution is not active. The exception is only raised
when graph execution is active, for example inside a `@tf.function` with
AutoGraph turned off. It can be caused by using a `tf.Tensor` value as:

  * the condition of an `if` or `while` statement: `if <tensor>:`
  * the argument in a logical expression: `tensor and another_tensor`
  * the argument to the `bool` built-in: `bool(tensor)`

Note: These operations are allowed when executing eagerly.

Within the context of AutoGraph, it usually indicates eager-style control
flow that has not been converted by AutoGraph, for any reason.

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
