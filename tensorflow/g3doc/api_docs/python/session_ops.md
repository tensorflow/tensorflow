<!-- This file is machine generated: DO NOT EDIT! -->

# Tensor Handle Operations

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Tensor Handle Operations.

TensorFlow provides several operators that allows the user to keep tensors
"in-place" across run calls.

- - -

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


- - -

### `tf.get_session_tensor(handle, dtype, name=None)` {#get_session_tensor}

Get the tensor of type `dtype` by feeding a tensor handle.

This is EXPERIMENTAL and subject to change.

Get the value of the tensor from a tensor handle. The tensor
is produced in a previous run() and stored in the state of the
session.

##### Args:


*  <b>`handle`</b>: The string representation of a persistent tensor handle.
*  <b>`dtype`</b>: The type of the output tensor.
*  <b>`name`</b>: Optional name prefix for the return tensor.

##### Returns:

  A pair of tensors. The first is a placeholder for feeding a
  tensor handle and the second is the tensor in the session state
  keyed by the tensor handle.


*  <b>`Example`</b>: 

```python
c = tf.mul(a, b)
h = tf.get_session_handle(c)
h = sess.run(h)

p, a = tf.get_session_tensor(h.handle, tf.float32)
b = tf.mul(a, 10)
c = sess.run(b, feed_dict={p: h.handle})
```


- - -

### `tf.delete_session_tensor(handle, name=None)` {#delete_session_tensor}

Delete the tensor for the given tensor handle.

This is EXPERIMENTAL and subject to change.

Delete the tensor of a given tensor handle. The tensor is produced
in a previous run() and stored in the state of the session.

##### Args:


*  <b>`handle`</b>: The string representation of a persistent tensor handle.
*  <b>`name`</b>: Optional name prefix for the return tensor.

##### Returns:

  A pair of graph elements. The first is a placeholder for feeding a
  tensor handle and the second is a deletion operation.


