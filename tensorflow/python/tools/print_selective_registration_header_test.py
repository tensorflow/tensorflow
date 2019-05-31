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
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.tools import selective_registration_header_lib

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

# AccumulateNV2 is included because it should be included in the header despite
# lacking a kernel (it's rewritten by AccumulateNV2RemovePass; see
# core/common_runtime/accumulate_n_optimizer.cc.
GRAPH_DEF_TXT_2 = """
  node: {
    name: "node_4"
    op: "BiasAdd"
    input: [ "none", "none" ]
    device: "/cpu:0"
    attr: { key: "T" value: { type: DT_FLOAT } }
  }
  node: {
    name: "node_5"
    op: "AccumulateNV2"
    attr: { key: "T" value: { type: DT_INT32 } }
    attr: { key  : "N" value: { i: 3 } }
  }

"""


class PrintOpFilegroupTest(test.TestCase):

  def setUp(self):
    _, self.script_name = os.path.split(sys.argv[0])

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

    ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)
    matmul_prefix = ''
    if test_util.IsMklEnabled():
      matmul_prefix = 'Mkl'

    self.assertListEqual(
        [
            ('AccumulateNV2', None),  #
            ('BiasAdd', 'BiasOp<CPUDevice, float>'),  #
            ('MatMul',
             matmul_prefix + 'MatMulOp<CPUDevice, double, false >'),  #
            ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, float, false >'),  #
            ('NoOp', 'NoOp'),  #
            ('Reshape', 'ReshapeOp'),  #
            ('_Recv', 'RecvOp'),  #
            ('_Send', 'SendOp'),  #
        ],
        ops_and_kernels)

    graphs[0].node[0].ClearField('device')
    graphs[0].node[2].ClearField('device')
    ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)
    self.assertListEqual(
        [
            ('AccumulateNV2', None),  #
            ('BiasAdd', 'BiasOp<CPUDevice, float>'),  #
            ('MatMul',
             matmul_prefix + 'MatMulOp<CPUDevice, double, false >'),  #
            ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, float, false >'),  #
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
    ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels(
        'rawproto', self.WriteGraphFiles(graphs), default_ops)

    header = selective_registration_header_lib.get_header_from_ops_and_kernels(
        ops_and_kernels, include_all_ops_and_kernels=True)
    self.assertListEqual(
        [
            '// This file was autogenerated by %s' % self.script_name,
            '#ifndef OPS_TO_REGISTER',  #
            '#define OPS_TO_REGISTER',  #
            '#define SHOULD_REGISTER_OP(op) true',  #
            '#define SHOULD_REGISTER_OP_KERNEL(clz) true',  #
            '#define SHOULD_REGISTER_OP_GRADIENT true',  #
            '#endif'
        ],
        header.split('\n'))

    self.assertListEqual(
        header.split('\n'),
        selective_registration_header_lib.get_header(
            self.WriteGraphFiles(graphs), 'rawproto', default_ops).split('\n'))

  def testGetSelectiveHeader(self):
    default_ops = ''
    graphs = [text_format.Parse(GRAPH_DEF_TXT_2, graph_pb2.GraphDef())]

    expected = '''// This file was autogenerated by %s
#ifndef OPS_TO_REGISTER
#define OPS_TO_REGISTER

    namespace {
      constexpr const char* skip(const char* x) {
        return (*x) ? (*x == ' ' ? skip(x + 1) : x) : x;
      }

      constexpr bool isequal(const char* x, const char* y) {
        return (*skip(x) && *skip(y))
                   ? (*skip(x) == *skip(y) && isequal(skip(x) + 1, skip(y) + 1))
                   : (!*skip(x) && !*skip(y));
      }

      template<int N>
      struct find_in {
        static constexpr bool f(const char* x, const char* const y[N]) {
          return isequal(x, y[0]) || find_in<N - 1>::f(x, y + 1);
        }
      };

      template<>
      struct find_in<0> {
        static constexpr bool f(const char* x, const char* const y[]) {
          return false;
        }
      };
    }  // end namespace
    constexpr const char* kNecessaryOpKernelClasses[] = {
"BiasOp<CPUDevice, float>",
};
#define SHOULD_REGISTER_OP_KERNEL(clz) (find_in<sizeof(kNecessaryOpKernelClasses) / sizeof(*kNecessaryOpKernelClasses)>::f(clz, kNecessaryOpKernelClasses))

constexpr inline bool ShouldRegisterOp(const char op[]) {
  return false
     || isequal(op, "AccumulateNV2")
     || isequal(op, "BiasAdd")
  ;
}
#define SHOULD_REGISTER_OP(op) ShouldRegisterOp(op)

#define SHOULD_REGISTER_OP_GRADIENT false
#endif''' % self.script_name

    header = selective_registration_header_lib.get_header(
        self.WriteGraphFiles(graphs), 'rawproto', default_ops)
    print(header)
    self.assertListEqual(expected.split('\n'), header.split('\n'))


if __name__ == '__main__':
  test.main()
