from __future__ import division, print_function

from numpy.random import random

from tensorflow.contrib.min_quantize.quantize_lib import feeds_of_graph, quantize_graph_def
from tensorflow.contrib.min_quantize.obfuscate_lib import obfuscate_quantized_graph
from tensorflow.python.client.session import Session
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.platform.googletest import main


class QuantizeTest(TestCase):
  def setUp(self):
    super(QuantizeTest, self).setUp()
    with Session(graph=Graph()) as session:
      with session.graph.as_default():
        # initial [-1, 1) random matrix
        x = constant(2 * random((1, 4096)) - 1, dtype=float32, name='c1')
        y = constant(2 * random((4096, 1)) - 1, dtype=float32, name='c2')
        # matmul to scalar
        z = matmul(x, y, name='c3')
      self._desire_z = session.run(z)
      self._quantized_raw = quantize_graph_def(session.graph_def, only='raw')
      self._quantized_simple = quantize_graph_def(session.graph_def, only='simple')
      self._quantized_full = quantize_graph_def(session.graph_def)

  def testRaw(self):
    self.assertAllClose(self._exec(self._quantized_raw), self._desire_z)

  def testSimple(self):
    self.assertAllClose(self._exec(self._quantized_simple), self._desire_z, 0.01, 0.01)

  def testFull(self):
    self.assertAllClose(self._exec(self._quantized_full), self._desire_z, 0.01, 0.01)

  def testFullIsBetter(self):
    self.assertLess(abs(self._exec(self._quantized_full) - self._desire_z),
                    abs(self._exec(self._quantized_simple) - self._desire_z))

  def testQuantize(self):
    graph, mapping = obfuscate_quantized_graph(self._quantized_full, keeps=['c3'])
    self.assertAllClose(self._exec(self._quantized_full), self._exec(graph))
    self.assertEqual({'a', 'b', 'c3'}, set(mapping.values()))

  @staticmethod
  def _exec(g):
    with Session(graph=Graph()) as session:
      with session.graph.as_default():
        import_graph_def(g.graph, name='')
      feeds = feeds_of_graph(g)
      return session.run('c3:0', feed_dict=feeds)


if __name__ == '__main__':
  main()
