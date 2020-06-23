# tf.experimental.numpy

This module provides a subset of numpy API, built on top of TensorFlow
operations. APIs are based on numpy 1.16 version.

The set of supported APIs may be expanded over time. Also future releases may
change the baseline version of numpy API being supported. A list of some
systematic differences with numpy are listed later in the "Differences with
Numpy" section.

## Getting Started

```python
import tensorflow as tf
from tf.experimental import numpy as np
print(np.ones([2,1]) + np.ones([1, 2]))
```

## Types

The module provide an `ndarray` class which wraps an immutable `tf.Tensor`.
Additional functions are provided which accept array-like objects. Here
array-like objects includes `ndarrays` as defined by this module, as well as
`tf.Tensor`, in addition to types accepted by `numpy`.

A subset of `numpy` dtypes are supported. Type promotion follows numpy
semantics.

```python
print(np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8))
```

## Interop

The numpy API calls can be interleaved with TensorFlow calls without incurring
Tensor data copies. This is true even if the `ndarray` or `tf.Tensor` is placed
on a non-CPU device.

Additionally, one could put these calls in a `with tf.GradientTape()` context to
compute gradients through the numpy API calls. Similarly, code vectorization can
be done using `tf.vectorized_map()`.

In general, the expected behavior should be on par with that of code involving
`tf.Tensor` and running stateless TensorFlow functions on them.

```python
np.sum(np.ones([1, 2]) + tf.ones([2, 1]))
```

## Array Interface

The `ndarray` class implements the `__array__` interface. This should allow
these objects to be passed into contexts that expect a `numpy` or array-like
object (e.g. matplotlib).

```python
import numpy as onp
onp.sum(np.ones([1, 2]) + onp.ones([2, 1]))
```

## Device Support

Given that `ndarray` and functions wrap TensorFlow constructs, the code will
have GPU and TPU support on par with TensorFlow. Also the code can be wrapped
with `tf.function` and XLA compiled. Device placement can be controlled by using
`with tf.device` scopes.

```python
with tf.device("GPU:0"):
  x = np.ones([1, 2])
print(tf.convert_to_tensor(x).device)
```

## Graph and Eager Modes

Eager mode execution should typically match numpy semantics of executing
op-by-op. However the same code can be executed in graph mode, by putting it
inside a `tf.function`. The function body can contain numpy code, and the inputs
can be ndarray as well.

```python
@tf.function
def f(x, y):
  return np.sum(x + y)

f(np.ones([1, 2]), tf.ones([2, 1]))
```

Note that this can change behavior of certain operations since symbolic
execution may not have information that is computed during runtime.

Some differences are:

*   Shapes can be incomplete or unknown. This means that `ndarray.shape`,
    `ndarray.size` and `ndarray.ndim` can return `ndarray` objects instead of
    returning integer (or tuple of integer) values.
*   Python control flow based on `ndarray` values may not work and may have to
    be rewritten to use `tf.cond` or `tf.while_loop`. Note that autograph
    conversion as part of `tf.function` should still work.
*   `__len__`, `__iter__` and `__index__` properties of `ndarray` may similarly
    not work in graph mode.

## Mutation and Variables

`ndarrays` currently wrap immutable `tf.Tensor`. Also currently mutation
operations like slice assigns are not supported. This may change in the future.

There is currently no explict construct on par with `tf.Variable`. However one
can directly construct a `tf.Variable` and use that with the numpy APIs in this
module. See section on Interop.

## Differences with Numpy

Here is a non-exhaustive list of differences:

*   Not all dtypes are currently supported. e.g. `np.float96`, `np.float128`.
    `np.object`, `np.str`, `np.recarray` types are not supported.
*   `ndarray` storage is in C order only. Fortran order, views, stride_tricks
    are not supported.
*   Only a subset of functions and modules are supported. This set would be
    expanded over time. For supported functions, some arguments or argument
    values may not be supported. This differences are listed in the function
    comments.
*   Buffer mutation is currently not supported. `ndarrays` wrap immutable
    tensors. This means that output buffer arguments (e..g `out` in ufuncs) are
    not supported
*   full `ufunc` support is not provided.
*   Numpy C API is not supported. Numpy's Cython and Swig integration are not
    supported.
