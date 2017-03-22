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
"""Definition of XLA test case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import random
import re

import numpy as np

from tensorflow.contrib.compiler import jit
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

FLAGS = flags.FLAGS

flags.DEFINE_string('test_device', None,
                    'Tensorflow device on which to place operators under test')
flags.DEFINE_string('types', None, 'Types to test. Comma-separated list.')
flags.DEFINE_string('disabled_manifest', None,
                    'Path to a file with a list of tests that should not run.')


class XLATestCase(test.TestCase):
  """XLA test cases are parameterized test cases."""

  def __init__(self, method_name='runTest'):
    super(XLATestCase, self).__init__(method_name)
    self.device = FLAGS.test_device
    self.has_custom_call = (self.device == 'XLA_CPU')
    self.all_tf_types = [
        dtypes.DType(types_pb2.DataType.Value(name))
        for name in FLAGS.types.split(',')
    ]
    self.all_types = [dtype.as_numpy_dtype for dtype in self.all_tf_types]
    self.int_types = [
        dtype.as_numpy_dtype for dtype in self.all_tf_types if dtype.is_integer
    ]
    self.float_types = [
        dtype.as_numpy_dtype for dtype in self.all_tf_types if dtype.is_floating
    ]
    self.numeric_types = self.int_types + self.float_types

    # Parse the manifest file, if any, into a regex identifying tests to
    # disable
    self.disabled_regex = None
    if FLAGS.disabled_manifest is not None:
      comments_re = re.compile('#.*$')
      manifest_file = open(FLAGS.disabled_manifest, 'r')
      lines = manifest_file.read().splitlines()
      lines = [comments_re.sub('', l).strip() for l in lines]
      self.disabled_regex = re.compile('|'.join(lines))
      manifest_file.close()

  def setUp(self):
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    if self.disabled_regex is not None and self.disabled_regex.match(name):
      logging.info('Disabled test case: %s', name)
      self.skipTest('{} is disabled by manifest.'.format(name))
      return
    logging.info('Start test case: %s', name)

    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)

  def tearDown(self):
    logging.info('End test case: %s', self._testMethodName)

  @contextlib.contextmanager
  def test_session(self):
    """Custom implementation of test_session() for XLA tests.

    We override the standard Tensorflow test_session() since it is too
    specific to CPU and GPU tests. In particular, we want to disable soft
    placement and explicitly assign ops to devices under test.

    Yields:
      A session to use when running a test case.
    """
    graph = ops.Graph()
    with session.Session(graph=graph) as sess, graph.as_default():
      yield sess

  @contextlib.contextmanager
  def test_scope(self):
    """Test scope that runs tests on a Tensorflow/XLA device.

    Uses a compilation_scope() to mark operators to compile.

    Yields:
      A scope to apply to the operators under test.
    """
    with ops.device('device:{}:0'.format(self.device)):
      yield


def Benchmark(tf_bench,
              builder_fn,
              use_xla_jit,
              device,
              separate_compiled_gradients=False):
  """Build a graph and run benchmarks against it, with or without XLA.

  Args:
    tf_bench: An instance of tf.test.Benchmark, used to run the benchmark.
    builder_fn: A function that builds a graph when invoked, and returns
        (name, fetches), where name is the name of the test, and fetches
        is a list of tensors to fetch as output.
    use_xla_jit: If true compile with the XLA JIT, otherwise use regular TF.
    device: The tensorflow device to run on, e.g. "cpu", "gpu".
    separate_compiled_gradients: If true put each gradient subgraph into a
      separate compilation scope. This gives fine-grained control over which
      portions of the graph will be compiled as a single unit. Compiling
      gradients separately may yield better performance for some graphs.
      The scope is named based on the scope of the forward computation as well
      as the name of the gradients. As a result, the gradients will be compiled
      in a scope that is separate from both the forward computation, and from
      other gradients.
  """

  with ops.Graph().as_default():
    name = None
    targets = []
    with ops.device(device):
      fetches = []
      jit_scope = jit.experimental_jit_scope
      with jit_scope(
          compile_ops=use_xla_jit,
          separate_compiled_gradients=separate_compiled_gradients):
        name, fetches = builder_fn()

      # We only want to benchmark the operations themselves, and not the data
      # transfer of the result(s).  Non-compiled identity ops ensure XLA
      # doesn't know we're dropping the results, otherwise it might compile
      # away the entire computation.
      for fetch in fetches:
        targets.append(array_ops.identity(fetch).op)

    config = config_pb2.ConfigProto(allow_soft_placement=True)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      xla = 'xla_' if use_xla_jit else ''
      tf_bench.run_op_benchmark(
          sess, targets, name='%s_%s%s' % (name, xla, device))
