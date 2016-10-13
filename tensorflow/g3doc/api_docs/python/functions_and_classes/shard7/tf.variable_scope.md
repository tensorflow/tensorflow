### `tf.variable_scope(name_or_scope, default_name=None, values=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, reuse=None, dtype=None)` {#variable_scope}

Returns a context manager for defining ops that creates variables (layers).

This context manager validates that the (optional) `values` are from
the same graph, ensures that graph is the default graph, and pushes a
name scope and a variable scope.

If `name_or_scope` is not None, it is used as is. If `scope` is None, then
`default_name` is used.  In that case, if the same name has been previously
used in the same scope, it will made unique be appending `_N` to it.

Variable scope allows to create new variables and to share already created
ones while providing checks to not create or share by accident. For details,
see the [Variable Scope How To](../../how_tos/variable_scope/index.md),
here we present only a few basic examples.

Simple example of how to create a new variable:

```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
```

Basic example of sharing a variable:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

Sharing a variable by capturing a scope and setting reuse:

```python
with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

To prevent accidental sharing of variables, we raise an exception when
getting an existing variable in a non-reusing scope.

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").
```

Similarly, we raise an exception when trying to get a variable that
does not exist in reuse mode.

```python
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
```

Note that the `reuse` flag is inherited: if we open a reusing scope,
then all its sub-scopes become reusing as well.

##### Args:


*  <b>`name_or_scope`</b>: `string` or `VariableScope`: the scope to open.
*  <b>`default_name`</b>: The default name to use if the `name_or_scope` argument is
    `None`, this name will be uniquified. If name_or_scope is provided it
    won't be used and therefore it is not required and can be None.
*  <b>`values`</b>: The list of `Tensor` arguments that are passed to the op function.
*  <b>`initializer`</b>: default initializer for variables within this scope.
*  <b>`regularizer`</b>: default regularizer for variables within this scope.
*  <b>`caching_device`</b>: default caching device for variables within this scope.
*  <b>`partitioner`</b>: default partitioner for variables within this scope.
*  <b>`custom_getter`</b>: default custom getter for variables within this scope.
*  <b>`reuse`</b>: `True` or `None`; if `True`, we go into reuse mode for this scope as
    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
*  <b>`dtype`</b>: type of variables created in this scope (defaults to the type
    in the passed scope, or inherited from parent scope).

##### Returns:

  A scope that can be to captured and reused.

##### Raises:


*  <b>`ValueError`</b>: when trying to reuse within a create scope, or create within
    a reuse scope, or if reuse is not `None` or `True`.
*  <b>`TypeError`</b>: when the types of some arguments are not appropriate.

