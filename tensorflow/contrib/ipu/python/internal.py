# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Graphcore utility operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from functools import wraps

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.platform import tf_logging as logging


def remap(x, name=None):
  """Clone and map the input linearly across the IPU.

  Args:
    x: The tensor to remap.
    name: Optional op name.

  Returns:
    A `Tensor` which is has been linearly mapped across the IPU.
  """

  logging.warning("remap is a Graphcore internal op")
  return gen_poputil_ops.ipu_remap(x, name=name)


def print_tensor(input, name=None):
  """Print the specified input.

  Args:
    input: The tensor to print.
    name: Optional op name.

  Returns:
    An operator that prints the specified input to the standard error. For the
    tensor to be printed one must either return it as part of their XLA function
    which is consumed by ipu_compiler.compile, or include the returned op in the
    input to session.run, or use the operator as a control dependency for
    executed ops by specifying with tf.control_dependencies([print_op]).

  Examples:
    1. Returning the print operation as part of the XLA function:

    .. code-block:: python

       import tensorflow as tf

       from tensorflow.contrib.ipu import internal
       from tensorflow.contrib.ipu import ops

       def my_net(v):
         print_op = internal.print_tensor(v)
         v = v + 1
         return v, print_op

       with ops.ipu_scope("/device:IPU:0"):
         res = ipu_compiler.compile(my_net, inputs=[v])

       ...
       ...

     2. Including the print operation in session.run:

     .. code-block:: python

       import numpy as np
       import tensorflow as tf

       from tensorflow.contrib.ipu import internal
       from tensorflow.contrib.ipu import ops

       with ops.ipu_scope("/device:IPU:0"):
         pa = tf.placeholder(np.float32, [2, 2], name="a")
         print_op = internal.print_tensor(pa)
         x = pa + 1

       with tf.Session() as session:
        result = session.run([x, print_op], feed_dict={pa : np.ones([2, 2])})

       ...
       ...

     3. Using control dependencies:

     .. code-block:: python

       import numpy as np
       import tensorflow as tf

       from tensorflow.contrib.ipu import internal
       from tensorflow.contrib.ipu import ops

       with ops.ipu_scope("/device:IPU:0"):
         pa = tf.placeholder(np.float32, [2, 2], name="a")
         print_op = internal.print_tensor(pa)
         with tf.control_dependencies([print_op]):
           x = pa + 1

       with tf.Session() as session:
        result = session.run(x, feed_dict={pa : np.ones([2, 2])})

       ...
       ...

  """

  return gen_poputil_ops.ipu_print_tensor(input, name=name)
