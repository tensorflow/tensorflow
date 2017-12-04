from __future__ import absolute_import, division, print_function

from tensorflow.contrib.min_quantize.obfuscate_lib import obfuscate_graph_def
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.platform.googletest import main


class ObfuscateTest(TestCase):
  def setUp(self):
    super(ObfuscateTest, self).setUp()
    self._graph = Graph()
    with self._graph.as_default():
      constant([1], dtype=int32, name='c1')
      constant([1], dtype=int32, name='c2')
      constant([1], dtype=int32, name='c3')

  def testObfuscate(self):
    ng, mp = obfuscate_graph_def(self._graph.as_graph_def())
    self.assertEqual(set(mp.values()), {'a', 'b', 'c'})

  def testKeep(self):
    ng, mp = obfuscate_graph_def(self._graph.as_graph_def(), ['c1', ('c2', 'd2')])
    self.assertEqual(set(mp.values()), {'a', 'c1', 'd2'})


if __name__ == '__main__':
  main()
