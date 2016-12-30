<!-- This file is machine generated: DO NOT EDIT! -->

# Wraps python functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Script Language Operators.

TensorFlow provides allows you to wrap python/numpy functions as
TensorFlow operators.

- - -

### `tf.py_func(func, inp, Tout, stateful=True, name=None)` {#py_func}

Wraps a python function and uses it as a TensorFlow op.

Given a python function `func`, which takes numpy arrays as its
inputs and returns numpy arrays as its outputs, wrap this function as an
operation in a TensorFlow graph. The following snippet constructs a simple
TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
in the graph:

```python
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32)
y = tf.py_func(my_func, [inp], tf.float32)
```

**N.B.** The `tf.py_func()` operation has the following known limitations:

* The body of the function (i.e. `func`) will not be serialized in a
  `GraphDef`. Therefore, you should not use this function if you need to
  serialize your model and restore it in a different environment.

* The operation must run in the same address space as the Python program
  that calls `tf.py_func()`. If you are using distributed TensorFlow, you
  must run a `tf.train.Server` in the same process as the program that calls
  `tf.py_func()` and you must pin the created operation to a device in that
  server (e.g. using `with tf.device():`).

##### Args:


*  <b>`func`</b>: A Python function, which accepts a list of NumPy `ndarray` objects
    having element types that match the corresponding `tf.Tensor` objects
    in `inp`, and returns a list of `ndarray` objects (or a single `ndarray`)
    having element types that match the corresponding values in `Tout`.
*  <b>`inp`</b>: A list of `Tensor` objects.
*  <b>`Tout`</b>: A list or tuple of tensorflow data types or a single tensorflow data
    type if there is only one, indicating what `func` returns.
*  <b>`stateful`</b>: (Boolean.) If True, the function should be considered stateful.
    If a function is stateless, when given the same input it will return the
    same output and have no observable side effects. Optimizations such as
    common subexpression elimination are only performed on stateless
    operations.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `Tensor` or a single `Tensor` which `func` computes.


