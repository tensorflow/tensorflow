### `tf.contrib.framework.deprecated_args(date, instructions, *deprecated_arg_names_or_tuples)` {#deprecated_args}

Decorator for marking specific function arguments as deprecated.

This decorator logs a deprecation warning whenever the decorated function is
called with the deprecated argument. It has the following format:

  Calling <function> (from <module>) with <arg> is deprecated and will be
  removed after <date>. Instructions for updating:
    <instructions>

<function> will include the class name if it is a method.

It also edits the docstring of the function: ' (deprecated arguments)' is
appended to the first line of the docstring and a deprecation notice is
prepended to the rest of the docstring.

##### Args:


*  <b>`date`</b>: String. The date the function is scheduled to be removed. Must be
    ISO 8601 (YYYY-MM-DD).
*  <b>`instructions`</b>: String. Instructions on how to update code using the
    deprecated function.
*  <b>`*deprecated_arg_names_or_tuples`</b>: String. or 2-Tuple(String,
    [ok_vals]).  The string is the deprecated argument name.
    Optionally, an ok-value may be provided.  If the user provided
    argument equals this value, the warning is suppressed.

##### Returns:

  Decorated function or method.

##### Raises:


*  <b>`ValueError`</b>: If date is not in ISO 8601 format, instructions are
    empty, the deprecated arguments are not present in the function
    signature, or the second element of a deprecated_tuple is not a
    list.

