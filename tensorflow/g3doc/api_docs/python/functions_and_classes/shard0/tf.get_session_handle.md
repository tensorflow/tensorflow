### `tf.get_session_handle(data, name=None)` {#get_session_handle}

Return the handle of `data`.

This is EXPERIMENTAL and subject to change.

Keep `data` "in-place" in the runtime and create a handle that can be
used to retrieve `data` in a subsequent run().

Combined with `get_session_tensor`, we can keep a tensor produced in
one run call in place, and use it as the input in a future run call.

##### Args:


*  <b>`data`</b>: A tensor to be stored in the session.
*  <b>`name`</b>: Optional name prefix for the return tensor.

##### Returns:

  A scalar string tensor representing a unique handle for `data`.

##### Raises:


*  <b>`TypeError`</b>: if `data` is not a Tensor.


*  <b>`Example`</b>: 

```python
c = tf.mul(a, b)
h = tf.get_session_handle(c)
h = sess.run(h)

p, a = tf.get_session_tensor(h.handle, tf.float32)
b = tf.mul(a, 10)
c = sess.run(b, feed_dict={p: h.handle})
```

