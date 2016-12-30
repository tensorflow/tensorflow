# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for print_selective_registration_header."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.tools import print_selective_registration_header

# Note that this graph def is not valid to be loaded - its inputs are not
# assigned correctly in all cases.
GRAPH_DEF_TXT = """
  node: {
    name: "node_1"
    op: "Reshape"
    input: [ "none", "none" ]
    device: "/cpu:0"
    attr: { key: "T" value: { type: DT_FLOAT } }
  }
  node: {
    name: "node_2"
    op: "MatMul"
    input: [ "none", "none" ]
    device: "/cpu:0"
    attr: { key: "T" value: { type: DT_FLOAT } }
    attr: { key: "transpose_a" value: { b: false } }
    attr: { key: "transpose_b" value: { b: false } }
  }
  node: {
    name: "node_3"
    op: "MatMul"
    input: [ "none", "none" ]
    device: "/cpu:0"
    attr: { key: "T" value: { type: DT_DOUBLE } }
    attr: { key: "transpose_a" value: { b: false } }
    attr: { key: "transpose_b" value: { b: false } }
  }
"""

GRAPH_DEF_TXT_2 = """
  node: {
    name: "node_4"
    op: "BiasAdd"
    input: [ "none", "none" ]
    device: "/cpu:0"
    attr: { key: "T" value: { type: DT_FLOAT } }
  }

"""


class PrintOpFilegroupTest(test.TestCase):

  def WriteGraphFiles(self, graphs):
    fnames = []
    for i, graph in enumerate(graphs):
      fname = os.path.join(self.get_temp_dir(), 'graph%s.pb' % i)
      with gfile.GFile(fname, 'wb') as f:
        f.write(graph.SerializeToString())
      fnames.append(fname)
    return fnames

  def testGetOps(self):
    default_ops = 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'
    graphs = [
        text_format.Parse(d, graph_pb2.GraphDef())
        for d in [GRAPH_DEF_TXT, GRAPH_DEF_TXT_2]
    ]

    ops_and_kernels = print_selective_registration_header.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)
    self.assertListEqual(
        [
            ('BiasAdd', 'BiasOp<CPUDevice, float>'),  #
            ('MatMul', 'MatMulOp<CPUDevice, double, false >'),  #
            ('MatMul', 'MatMulOp<CPUDevice, float, false >'),  #
            ('NoOp', 'NoOp'),  #
            ('Reshape', 'ReshapeOp'),  #
            ('_Recv', 'RecvOp'),  #
            ('_Send', 'SendOp'),  #
        ],
        ops_and_kernels)

    graphs[0].node[0].ClearField('device')
    graphs[0].node[2].ClearField('device')
    ops_and_kernels = print_selective_registration_header.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)
    self.assertListEqual(
        [
            ('BiasAdd', 'BiasOp<CPUDevice, float>'),  #
            ('MatMul', 'MatMulOp<CPUDevice, double, false >'),  #
            ('MatMul', 'MatMulOp<CPUDevice, float, false >'),  #
            ('NoOp', 'NoOp'),  #
            ('Reshape', 'ReshapeOp'),  #
            ('_Recv', 'RecvOp'),  #
            ('_Send', 'SendOp'),  #
        ],
        ops_and_kernels)

  def testAll(self):
    default_ops = 'all'
    graphs = [
        text_format.Parse(d, graph_pb2.GraphDef())
        for d in [GRAPH_DEF_TXT, GRAPH_DEF_TXT_2]
    ]
    ops_and_kernels = print_selective_registration_header.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)

    header = print_selective_registration_header.get_header(ops_and_kernels,
                                                            default_ops)
    self.assertListEqual(
        [
            '#ifndef OPS_TO_REGISTER',  #
            '#define OPS_TO_REGISTER',  #
            '#define SHOULD_REGISTER_OP(op) true',  #
            '#define SHOULD_REGISTER_OP_KERNEL(clz) true',  #
            '#define SHOULD_REGISTER_OP_GRADIENT true',  #
            '#endif'
        ],
        header.split('\n'))


if __name__ == '__main__':
  test.main()
