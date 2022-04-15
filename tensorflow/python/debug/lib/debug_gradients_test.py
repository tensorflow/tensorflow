# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for debug_gradients module."""

import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_gradients
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


@test_util.run_v1_only("Sessions are not available in TF 2.x")
class IdentifyGradientTest(test_util.TensorFlowTestCase):

  def setUp(self):
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    self.sess = session.Session(config=config)
    with self.sess.as_default():
      self.u = variables.Variable(2.0, name="u")
      self.v = variables.Variable(3.0, name="v")
      self.w = math_ops.multiply(self.u.value(), self.v.value(), name="w")

  def tearDown(self):
    ops.reset_default_graph()
    debug_gradients.clear_gradient_debuggers()

  def testIdentifyGradientGivesCorrectTensorObjectWithoutContextManager(self):
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor's name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testIdentifyGradientGivesCorrectTensorObjectWithTfGradients(self):
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    with grad_debugger:
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor's name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testCallingIdentifyGradientTwiceWithTheSameGradientsDebuggerErrors(self):
    grad_debugger = debug_gradients.GradientsDebugger()
    grad_debugger.identify_gradient(self.w)
    with self.assertRaisesRegex(ValueError,
                                "The graph already contains an op named .*"):
      grad_debugger.identify_gradient(self.w)

  def testIdentifyGradientWorksOnMultipleLosses(self):
    grad_debugger_1 = debug_gradients.GradientsDebugger()
    grad_debugger_2 = debug_gradients.GradientsDebugger()

    y = math_ops.add(self.w, -1.0, name="y")
    debug_y = grad_debugger_1.identify_gradient(y)
    z1 = math_ops.square(debug_y, name="z1")

    debug_y = grad_debugger_2.identify_gradient(y)
    z2 = math_ops.sqrt(debug_y, name="z2")

    with grad_debugger_1:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z1)
    with grad_debugger_2:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z2)

    dz1_dy = grad_debugger_1.gradient_tensor(y)
    dz2_dy = grad_debugger_2.gradient_tensor(y)
    self.assertIsInstance(dz1_dy, ops.Tensor)
    self.assertIsInstance(dz2_dy, ops.Tensor)
    self.assertIsNot(dz1_dy, dz2_dy)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0**2, self.sess.run(z1))
    self.assertAllClose(5.0**0.5, self.sess.run(z2))
    self.assertAllClose(2.0 * 5.0, self.sess.run(dz1_dy))
    self.assertAllClose(0.5 * (5.0**-0.5), self.sess.run(dz2_dy))

  def testIdentifyGradientRaisesLookupErrorForUnknownXTensor(self):
    grad_debugger_1 = debug_gradients.GradientsDebugger()
    grad_debugger_2 = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger_1.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    # There are >1 gradient debuggers registered, and grad_debugger is not used
    # as a context manager here, so the gradient w.r.t. self.w will not be
    # registered.
    gradients_impl.gradients(y, [self.u, self.v])

    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "):
      grad_debugger_1.gradient_tensor(self.w)
    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "):
      grad_debugger_2.gradient_tensor(self.w)

  def testIdentifyGradientRaisesTypeErrorForNonTensorOrTensorNameInput(self):
    grad_debugger = debug_gradients.GradientsDebugger()
    with self.assertRaisesRegex(
        TypeError,
        r"x_tensor must be a str or tf\.Tensor or tf\.Variable, but instead "
        r"has type .*Operation.*"):
      grad_debugger.gradient_tensor(variables.global_variables_initializer())

  def testIdentifyGradientTensorWorksWithGradientDescentOptimizer(self):
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    with grad_debugger:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(y)

    self.sess.run(variables.global_variables_initializer())

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsByXTensorNamesWorks(self):
    y = math_ops.add(self.w, -1.0, name="y")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_tensor_names().
    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "w:0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    w_grad = grad_debugger.gradient_tensor("w:0")
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsByXTensorNamesWorksWithoutContextManager(self):
    y = math_ops.add(self.w, -1.0, name="y")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_tensor_names().
    grad_debugger = debug_gradients.GradientsDebugger()
    grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "w:0$")
    grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    w_grad = grad_debugger.gradient_tensor("w:0")
    self.assertIsInstance(w_grad, ops.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsWorksOnRefTensor(self):
    y = math_ops.add(self.w, -1.0, name="y")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "u:0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.assertIs(u_grad, grad_debugger.gradient_tensor("u:0"))

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))
    self.assertAllClose(3.0, self.sess.run(
        grad_debugger.gradient_tensor("u:0")))

  def testWatchGradientsWorksOnMultipleTensors(self):
    y = math_ops.add(self.w, -1.0, name="y")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph,
                                                       "(u|w):0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]

    self.assertEqual(2, len(grad_debugger.gradient_tensors()))
    self.assertIs(u_grad, grad_debugger.gradient_tensor("u:0"))
    self.assertIsInstance(grad_debugger.gradient_tensor("w:0"), ops.Tensor)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(1.0, self.sess.run(
        grad_debugger.gradient_tensor("w:0")))
    self.assertAllClose(3.0, self.sess.run(
        grad_debugger.gradient_tensor("u:0")))

  def testWatchGradientsByXTensorsWorks(self):
    y = math_ops.add(self.w, -1.0, name="foo/y")
    z = math_ops.square(y, name="foo/z")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_x_tensors().
    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(self.sess.graph,
                                                  [self.w, self.u, y]):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z)

    self.assertEqual(3, len(grad_debugger.gradient_tensors()))
    u_grad = grad_debugger.gradient_tensor(self.u)
    w_grad = grad_debugger.gradient_tensor(self.w)
    y_grad = grad_debugger.gradient_tensor(y)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(10.0, self.sess.run(y_grad))
    self.assertAllClose(10.0, self.sess.run(w_grad))
    self.assertAllClose(30.0, self.sess.run(u_grad))

  def testWatchGradientsByTensorCanWorkOnMultipleLosses(self):
    y = math_ops.add(self.w, -1.0, name="y")
    z1 = math_ops.square(y, name="z1")
    z2 = math_ops.sqrt(y, name="z2")

    grad_debugger_1 = debug_gradients.GradientsDebugger()
    with grad_debugger_1.watch_gradients_by_tensors(self.sess.graph, y):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z1)

    grad_debugger_2 = debug_gradients.GradientsDebugger()
    with grad_debugger_2.watch_gradients_by_tensors(self.sess.graph, y):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z2)

    dz1_dy = grad_debugger_1.gradient_tensor(y)
    dz2_dy = grad_debugger_2.gradient_tensor(y)
    self.assertIsInstance(dz1_dy, ops.Tensor)
    self.assertIsInstance(dz2_dy, ops.Tensor)
    self.assertIsNot(dz1_dy, dz2_dy)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0**2, self.sess.run(z1))
    self.assertAllClose(5.0**0.5, self.sess.run(z2))
    self.assertAllClose(2.0 * 5.0, self.sess.run(dz1_dy))
    self.assertAllClose(0.5 * (5.0**-0.5), self.sess.run(dz2_dy))

  def testGradientsValuesFromDumpWorks(self):
    y = math_ops.add(self.w, -1.0, name="y")
    z = math_ops.square(y, name="z")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(self.sess.graph,
                                                  [self.w, self.u, y]):
      train_op = gradient_descent.GradientDescentOptimizer(0.1).minimize(z)

    self.sess.run(variables.global_variables_initializer())

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    dump_dir = tempfile.mkdtemp()
    debug_url = "file://" + dump_dir
    debug_utils.watch_graph(run_options, self.sess.graph, debug_urls=debug_url)
    run_metadata = config_pb2.RunMetadata()
    self.assertAllClose(2.0, self.sess.run(self.u))
    self.sess.run(train_op, options=run_options, run_metadata=run_metadata)
    self.assertAllClose(-1.0, self.sess.run(self.u))

    dump = debug_data.DebugDumpDir(
        dump_dir, partition_graphs=run_metadata.partition_graphs)
    dump.set_python_graph(self.sess.graph)

    y_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, y, dump)
    self.assertEqual(1, len(y_grad_values))
    self.assertAllClose(10.0, y_grad_values[0])

    w_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, self.w, dump)
    self.assertEqual(1, len(w_grad_values))
    self.assertAllClose(10.0, w_grad_values[0])

    u_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, self.u, dump)
    self.assertEqual(1, len(u_grad_values))
    self.assertAllClose(30.0, u_grad_values[0])

    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "
        r"x-tensor v:0"):
      debug_gradients.gradient_values_from_dump(grad_debugger, self.v, dump)

    # Cleanup.
    file_io.delete_recursively(dump_dir)


if __name__ == "__main__":
  googletest.main()
