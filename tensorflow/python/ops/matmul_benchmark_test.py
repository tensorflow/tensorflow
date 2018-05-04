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
"""Tests for matmul_benchmark.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import matmul_benchmark
from tensorflow.python.platform import test as googletest
from tensorflow.python.platform import tf_logging


def BuildGraphTest(n, m, k, transpose_a, transpose_b, dtype):

  def Test(self):
    if not googletest.is_gpu_available():
      tf_logging.info("Skipping BuildGraphTest %s",
                      (n, m, k, transpose_a, transpose_b))
      return
    tf_logging.info("Testing BuildGraphTest %s",
                    (n, m, k, transpose_a, transpose_b))
    self._VerifyBuildGraph(n, m, k, transpose_a, transpose_b, dtype)

  return Test


def RunGraphTest(n, m, k, transpose_a, transpose_b, dtype):

  def Test(self):
    if not googletest.is_gpu_available():
      tf_logging.info("Skipping RunGraphTest %s",
                      (n, m, k, transpose_a, transpose_b))
      return
    tf_logging.info("Testing RunGraphTest %s",
                    (n, m, k, transpose_a, transpose_b))
    self._VerifyRunGraph(n, m, k, transpose_a, transpose_b, dtype)

  return Test


class MatmulBenchmarkTest(googletest.TestCase):

  def _StripNode(self, nd):
    snode = node_def_pb2.NodeDef(name=nd.name, op=nd.op, input=nd.input)
    if nd.device:
      snode.device = nd.device
    return snode

  def _StripGraph(self, gd):
    return graph_pb2.GraphDef(node=[self._StripNode(nd) for nd in gd.node])

  def _VerifyBuildGraph(self, n, m, k, transpose_a, transpose_b, dtype):
    graph = ops.Graph()
    with graph.as_default():
      matmul_benchmark.build_graph(googletest.gpu_device_name(), n, m, k,
                                   transpose_a, transpose_b, dtype)
      gd = graph.as_graph_def()
      dev = googletest.gpu_device_name()
      proto_expected = """
      node { name: "random_uniform/shape" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform/min" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform/max" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform/RandomUniform" op: "RandomUniform" input: "random_uniform/shape" device: \"""" + dev + """\" }
      node { name: "random_uniform/sub" op: "Sub" input: "random_uniform/max" input: "random_uniform/min" device: \"""" + dev + """\" }
      node { name: "random_uniform/mul" op: "Mul" input: "random_uniform/RandomUniform" input: "random_uniform/sub" device: \"""" + dev + """\" }
      node { name: "random_uniform" op: "Add" input: "random_uniform/mul" input: "random_uniform/min" device: \"""" + dev + """\" }
      node { name: "Variable" op: "VariableV2" device: \"""" + dev + """\" }
      node { name: "Variable/Assign" op: "Assign" input: "Variable" input: "random_uniform" device: \"""" + dev + """\" }
      node { name: "Variable/read" op: "Identity" input: "Variable" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/shape" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/min" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/max" op: "Const" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/RandomUniform" op: "RandomUniform" input: "random_uniform_1/shape" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/sub" op: "Sub" input: "random_uniform_1/max" input: "random_uniform_1/min" device: \"""" + dev + """\" }
      node { name: "random_uniform_1/mul" op: "Mul" input: "random_uniform_1/RandomUniform" input: "random_uniform_1/sub" device: \"""" + dev + """\" }
      node { name: "random_uniform_1" op: "Add" input: "random_uniform_1/mul" input: "random_uniform_1/min" device: \"""" + dev + """\" }
      node { name: "Variable_1" op: "VariableV2" device: \"""" + dev + """\" }
      node { name: "Variable_1/Assign" op: "Assign" input: "Variable_1" input: "random_uniform_1" device: \"""" + dev + """\" }
      node { name: "Variable_1/read" op: "Identity" input: "Variable_1" device: \"""" + dev + """\" }
      node { name: "MatMul" op: "MatMul" input: "Variable/read" input: "Variable_1/read" device: \"""" + dev + """\" }
      node { name: "group_deps" op: "NoOp" input: "^MatMul" device: \"""" + dev + """\" }
                       """
      self.assertProtoEquals(str(proto_expected), self._StripGraph(gd))

  def _VerifyRunGraph(self, n, m, k, transpose_a, transpose_b, dtype):
    benchmark_instance = matmul_benchmark.MatmulBenchmark()
    duration = benchmark_instance.run_graph(googletest.gpu_device_name(), n, m,
                                            k, transpose_a, transpose_b, 1,
                                            dtype)
    self.assertTrue(duration > 1e-6)


if __name__ == "__main__":
  dtypes = [np.float32, np.float64]
  index = 0
  for _dtype in dtypes:
    for _n, _m, (_transpose_a, _transpose_b) in itertools.product(
        [512, 1024], [1, 8, 16, 128], [(False, False), (True, False),
                                       (False, True)]):
      _k = _n
      setattr(MatmulBenchmarkTest, "testBuildGraph_" + str(index),
              BuildGraphTest(_n, _m, _k, _transpose_a, _transpose_b, _dtype))
      setattr(MatmulBenchmarkTest, "testRunGraph_" + str(index),
              RunGraphTest(_n, _m, _k, _transpose_a, _transpose_b, _dtype))
      index += 1
  googletest.main()
