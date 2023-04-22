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
"""Backend-independent tests for the Python XLA client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import absltest
import numpy as np

from tensorflow.compiler.xla.python import xla_client

# pylint: disable=g-import-not-at-top
try:
  import portpicker
except ImportError:
  portpicker = None
# pylint: enable=g-import-not-at-top

ops = xla_client.ops


class ShapeTest(absltest.TestCase):

  def testInvalidShapes(self):
    with self.assertRaisesRegex(RuntimeError, "invalid shape"):
      xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [-2, 4])

    with self.assertRaisesRegex(
        RuntimeError, "layout minor_to_major field contains 1 element.*"):
      xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [2, 4], [3])

    with self.assertRaisesRegex(
        RuntimeError, "layout minor_to_major field has out-of-bounds value.*"):
      xla_client.Shape.array_shape(xla_client.PrimitiveType.F32, [2, 4],
                                   [1, -1])


class ComputationPrinting(absltest.TestCase):

  def ExampleComputation(self):
    builder = xla_client.XlaBuilder("acomputation")
    p0 = ops.Parameter(builder, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    x = ops.Mul(p0, p1)
    ops.Add(x, x)
    return builder.build()

  def testComputationToHloText(self):
    computation = self.ExampleComputation()
    hlo_text = computation.as_hlo_text()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))

  def testComputationToHloGraph(self):
    computation = self.ExampleComputation()
    hlo_dot_graph = computation.as_hlo_dot_graph()
    self.assertTrue(hlo_dot_graph.startswith("digraph "))

  def testHloModuleToHloText(self):
    computation = self.ExampleComputation()
    hlo_text = computation.as_hlo_module().to_string()
    self.assertTrue(hlo_text.startswith("HloModule acomputation"))

  def testHloModuleToHloGraph(self):
    computation = self.ExampleComputation()
    hlo_dot_graph = xla_client._xla.hlo_module_to_dot_graph(
        computation.as_hlo_module())
    self.assertTrue(hlo_dot_graph.startswith("digraph "))


class ComputationHashTest(absltest.TestCase):

  def testHash(self):
    builder0 = xla_client.XlaBuilder("computation0")
    p0 = ops.Parameter(builder0, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder0, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    ops.Mul(p0, p1)
    computation0 = builder0.build()

    builder1 = xla_client.XlaBuilder("computation1")
    p0 = ops.Parameter(builder1, 0, xla_client.shape_from_pyval(np.float32(0)))
    p1 = ops.Parameter(builder1, 1,
                       xla_client.shape_from_pyval(np.zeros((4,), np.float32)))
    ops.Mul(p0, p1)
    computation1 = builder1.build()

    self.assertEqual(computation0.hash(), computation1.hash())


class AliasTest(absltest.TestCase):

  def testSetUpAlias(self):
    c = xla_client.XlaBuilder(self.id())
    p1 = ops.Parameter(
        c, 0,
        xla_client.shape_from_pyval(np.array(
            1.0, np.float32)).with_major_to_minor_layout_if_absent())
    p2 = ops.Parameter(
        c, 1,
        xla_client.shape_from_pyval(np.array(
            1.0, np.float32)).with_major_to_minor_layout_if_absent())
    out = ops.Add(p1, p2)
    c.setup_alias([], 0, [])
    c.build(out)


class ProfilerTest(absltest.TestCase):

  def testTraceMe(self):
    # TODO(phawkins): These tests just check that the TraceMe context manager
    # acts like a context manager and doesn't explode. Ideally we'd check that
    # the profiler saw the traceme too.
    with xla_client.profiler.TraceMe("test1"):
      pass
    with xla_client.profiler.TraceMe("test2", foo=123):
      pass
    with self.assertRaises(ValueError):
      with xla_client.profiler.TraceMe("test3"):
        raise ValueError("test")

  @unittest.skipIf(portpicker is None, "Test requires portpicker")
  def testStartServer(self):
    port = portpicker.pick_unused_port()
    server = xla_client.profiler.start_server(port)
    del server


if __name__ == "__main__":
  absltest.main()
