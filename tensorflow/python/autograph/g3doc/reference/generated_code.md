# AutoGraph reference

[Index](index.md)

## Generated code

For each converted function, AutoGraph creates a new function. The
loading mechanism is an implementation detail and may change, but the
generated function is generally a regular
[Python function](https://docs.python.org/3/reference/compound_stmts.html#function).
This function is typically executed internally by `@tf.function` to construct a
TensorFlow graph.

### Transformations made to the generated code

The generated code is a transformation of the input code. The transformations
are listed below. Any other elements are left unchanged.

Summary of transformations:

 * function calls are replaced with a wrapper:
   * `foo(args)` -> `ag__.converted_call(foo, args)`
 * `if`, `while` and `for` statements are replaced with function calls:
   * `if` -> `ag__.if_stmt`
   * `while` -> `ag__.while_stmt`
   * `for` -> `ag__.for_stmt`
 * `break`, `return`, and `continue` statements are replaced with equivalent
   `if` statements.
 * `and`, `or` and `not` operators are replaced with function calls:
   * `and` -> `ag__.and_`
   * `or` -> `ag__.or_`
   * `not` -> `ag__.not_`

The functions replacing control flow statements are very similar in form with
the corresponding control flow ops in TensorFlow.

### AutoGraph generates normal Python code

You can interact normally with the generated code. For example, you can use
the `inspect.getsourcefile` and `inspect.getsource`:

```
def f(a):
  ...

converted_f = tf.autograph.to_graph(f)
print(inspect.getsourcefile(converted_f))
```
```
/tmp/tmpm562wlj7.py
```

When using `@tf.function`, you can repeat the same steps using the function's
`python_function` attribute:

```
@tf.function
def f(a):
  ...

converted_f = tf.autograph.to_graph(f.python_function)
print(inspect.getsourcefile(converted_f))
```
```
/tmp/tmpm562wlj7.py
```

`tf.autograph.to_code` is a shortcut to obtain the generated code, and it's
equivalent with calling `inspect.getsource(tf.autograph.to_graph(f))`.

#### Recording diagnostic information: `tf.autograph.set_verbosity`

AutoGraph can log additional debug information. This is mostly used for filing
bugs, but can also be used to get an indication of whether a function is
converted successfully or not.

You can enable logging by calling `tf.autograph.set_verbosity(level)`. The
`level` argument varies from 0 to 10:

 * 0 - no logging
 * 3 - includes the generated code
 * 4 and above - extremely verbose logging

Caution: The information being logged includes source code as well as
data. Before sharing AutoGraph logs, make sure they don't contain any sensitive
information.

Alternatively, you can control the verbosity level using the environment
variable `AUTOGRAPH_VERBOSITY`.
