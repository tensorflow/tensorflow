# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""numpy_ops.

This module provides a subset of numpy API, built on top of TensorFlow
operations. APIs are based on numpy 1.16 version.

The set of supported APIs may be expanded over time. Also future releases may
change the baseline version of numpy API being supported. A list of some
systematic differences with numpy are listed later in the "Differences with
Numpy" section.

Types
-----

The module provide an `ndarray` class which wraps an immutable `tf.Tensor`.
Additional functions are provided which accept array-like objects. Here
array-like objects includes `ndarrays` as defined by this module, as well as
`tf.Tensor`, in addition to types accepted by `numpy`.

A subset of `numpy` dtypes are supported, along with `tf.bfloat16`.
Additionally, support is provided for selecting the default float type
(`np.float32` vs `np.float64`) given that some applications may prefer lower
precision.

Device Support
-------------

Given that `ndarray` and functions wrap TensorFlow constructs, the code will
have GPU and TPU support on par with TensorFlow. Also the code can be wrapped
with `tf.function` and XLA compiled. Device placement can be controlled by using
`with tf.device` scopes.

Graph and Eager Modes
--------------------

Eager mode execution should typically match numpy semantics of executing
op-by-op. However the same code can be executed in graph mode, by putting it
inside a `tf.function`. This can change behavior of certain operations since
symbolic execution may not have information that is computed during runtime.

Some differences are:
  * Shapes can be incomplete or unknown. This means that `ndarray.shape`,
    `ndarray.size` and `ndarray.ndim` can return `ndarray` objects instead of
    returning integer (or tuple of integer) values.
  * Python control flow based on `ndarray` values may not work and may have to
    be rewritten to use `tf.cond` or `tf.while_loop`. Note that autograph
    conversion as part of `tf.function` should still work.
  * `__len__`, `__iter__` and `__index__` properties of `ndarray` may similarly
    not work in graph mode.

Mutation and Variables
---------------------

`ndarrays` currently wrap immutable `tf.Tensor`. Also currently mutation
operations like slice assigns are not supported. This may change in the future.

There is currently no explict construct on par with tf.Variable. However one can
directly construct a `tf.Variable` and use that with the numpy APIs in this
module. See section on Interop.

Interop
------

The numpy API calls can be interleaved with TensorFlow calls without incurring
Tensor data copies. This is true even if the `ndarray` or `tf.Tensor` is placed
on a non-CPU device.

Additionally, one could put these calls in a `with tf.GradientTape()` context to
compute gradients through the numpy API calls. Similarly, code vectorization can
be done using `tf.vectorized_map()`.

In general, the expected behavior should be on par with that of code involving
`tf.Tensor` and running stateless TensorFlow functions on them.

Array Interface
--------------

The `ndarray` class implements the `__array__ interface. This should allow these
objects to be passed into contexts that expect a `numpy` or array-like object
(e.g. matplotlib).


Differences with Numpy
---------------------

Here is a non-exhaustive list of differences:
  * Not all dtypes are currently supported. e.g. `np.float96`, `np.float128`.
    `np.object`, `np.str`, `np.recarray` types are not supported.
  * `ndarray` storage is in C order only. Fortran order, views, stride_tricks
    are not supported.
  * Only a subset of functions and modules are supported. This set would be
    expanded over time. For supported functions, some arguments or argument
    values may not be supported. This differences are listed in the function
    comments.
  * Buffer mutation is currently not supported. `ndarrays` wrap immutable
    tensors. This means that output buffer arguments (e..g `out` in ufuncs) are
    not supported
  * full `ufunc` support is not provided.
  * Numpy C API is not supported. Numpy's Cython and Swig integration are not
    supported.
"""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.numpy_ops import np_random as random
# pylint: disable=wildcard-import
from tensorflow.python.ops.numpy_ops.np_array_ops import *
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from tensorflow.python.ops.numpy_ops.np_dtypes import *
from tensorflow.python.ops.numpy_ops.np_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.ops.numpy_ops.np_utils import finfo
from tensorflow.python.ops.numpy_ops.np_utils import promote_types
from tensorflow.python.ops.numpy_ops.np_utils import result_type

# pylint: disable=redefined-builtin,undefined-variable
max = amax
min = amin
round = around
# pylint: enable=redefined-builtin,undefined-variable
