### `tf.contrib.framework.deprecated_arg_values(date, instructions, **deprecated_kwargs)` {#deprecated_arg_values}

Decorator for marking specific function argument values as deprecated.

This decorator logs a deprecation warning whenever the decorated function is
called with the deprecated argument values. It has the following format:

  Calling <function> (from <module>) with <arg>=<value> is deprecated and
  will be removed after <date>. Instructions for updating:
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
*  <b>`**deprecated_kwargs`</b>: The deprecated argument values.

##### Returns:

  Decorated function or method.

##### Raises:


*  <b>`ValueError`</b>: If date is not in ISO 8601 format, or instructions are empty.

