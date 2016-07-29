A TensorFlow `Session` for use in interactive contexts, such as a shell.

The only difference with a regular `Session` is that an `InteractiveSession`
installs itself as the default session on construction.
The methods [`Tensor.eval()`](../../api_docs/python/framework.md#Tensor.eval)
and [`Operation.run()`](../../api_docs/python/framework.md#Operation.run)
will use that session to run ops.

This is convenient in interactive shells and [IPython
notebooks](http://ipython.org), as it avoids having to pass an explicit
`Session` object to run ops.

For example:

```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()
```

Note that a regular session installs itself as the default session when it
is created in a `with` statement.  The common usage in non-interactive
programs is to follow that pattern:

```python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # We can also use 'c.eval()' here.
  print(c.eval())
```

- - -

#### `tf.InteractiveSession.__init__(target='', graph=None, config=None)` {#InteractiveSession.__init__}

Creates a new interactive TensorFlow session.

If no `graph` argument is specified when constructing the session,
the default graph will be launched in the session. If you are
using more than one graph (created with `tf.Graph()` in the same
process, you will have to use different sessions for each graph,
but each graph can be used in multiple sessions. In this case, it
is often clearer to pass the graph to be launched explicitly to
the session constructor.

##### Args:


*  <b>`target`</b>: (Optional.) The execution engine to connect to.
    Defaults to using an in-process engine. At present, no value
    other than the empty string is supported.
*  <b>`graph`</b>: (Optional.) The `Graph` to be launched (described above).
*  <b>`config`</b>: (Optional) `ConfigProto` proto used to configure the session.


- - -

#### `tf.InteractiveSession.close()` {#InteractiveSession.close}

Closes an `InteractiveSession`.


