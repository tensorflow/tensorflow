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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class JitCompileTest(test.TestCase):

  def testBasic(self):
    with ops.Graph().as_default() as g:

      def fn(x, a):
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  def testDerivative(self):
    def fn(x, a):
      return 2 * x + a

    with ops.Graph().as_default() as g:
      xla_func = def_function.function(fn, jit_compile=True)
      with backprop.GradientTape() as tape:
        inputs = array_ops.placeholder(dtypes.float32, [5])
        tape.watch(inputs)
        outputs = xla_func(inputs, 1)
      grads = tape.gradient(outputs, inputs)

    with session.Session(graph=g) as sess:
      grads_tensor = sess.run(grads, feed_dict={inputs: [1, 2, 2, 3, 3]})
      self.assertAllClose([2, 2, 2, 2, 2], grads_tensor)
      (forward, backward) = xla_func.get_concrete_function(
          inputs, 1)._delayed_rewrite_functions.forward_backward()

      # Check that the must-compile attribute gets correctly propagated to the
      # created derivatives.
      self.assertTrue(forward.definition.attr["_XlaMustCompile"])
      self.assertTrue(backward.function_def.attr["_XlaMustCompile"])

  def testBasicInt32(self):
    with ops.Graph().as_default() as g:

      def fn(x, a):
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.int32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  # Checking that we crash on an unsupported operation lets us test that the XLA
  # compiler was actually invoked.
  def testUnsupportedOps(self):
    with ops.Graph().as_default() as g:

      def fn(x):
        return array_ops.unique(x).y  # Unique is not supported by XLA

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Detected unsupported operations"):
        with session.Session(graph=g) as sess:
          sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})


if __name__ == "__main__":
  test.main()
