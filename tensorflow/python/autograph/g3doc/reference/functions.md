# AutoGraph reference

[Index](index.md)

## Functions and function calls

Typically, AutoGraph converts one function at a time. If a function calls other
functions, the called function will be converted recursively, as described
below.

### Function calls

AutoGraph rewrites all function calls with a special wrapper that may convert
the called function at runtime.

For example, the function call below:

```
f(x, y, z=1)
```

Is converted to code that schematically looks like this:

```
ag__.converted_call(f, ..., (x, y), {'z': 1}, ...)
```

All calls are rewritten, including calls to other types of callables, builtin
functions, etc.

If the originally called function is not converted, AutoGraph simply
forwards the call to it, so that the wrapper is functionally equivalent with
the original function call.

If the originally called function is converted, then the conversion is performed
first and the converted function is called instead.

Note: a caching mechanism prevents the same function from being converted
multiple times. This mechanism ensures that functions calls made with different
[global or free variables](https://docs.python.org/3/reference/executionmodel.html#binding-of-names)
are handled correctly.

#### Function conversion rules

The following types of functions are not converted:

*   functions already converted
*   functions defined in a allowlisted module (see autograph/core/config.py)
*   non-Python functions (such as native bindings)
*   `print`, `pdb.set_trace`, `ipdb.set_trace`
*   most built-in functions (exceptions are listed in
    autograph/operators/py_builtins.py)
*   constructors
*   functions without source code attached (prints a warning)(see
    [limitations](limitations.md))
*   generator functions (prints a warning)
*   iterator protocol methods (`__next__`, `__iter__`)
*   context manager methods (`__enter__`, `__exit__`)

When AutoGraph encounters a function that it cannot convert outside of this
list, it prints a warning.

### Nested functions

Functions nested inside a function converted by AutoGraph are converted
at the same time as the function containing them. If the nested function is
returned, a converted version of it is returned.
