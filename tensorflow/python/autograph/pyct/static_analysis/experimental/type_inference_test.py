# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for type_inference module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.experimental import type_inference
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

x = qual_names.from_str("x")
y = qual_names.from_str("y")
tensor_type = qual_names.from_str("ops.Tensor")

class TypeInferenceAnalyzerTestBase(test.TestCase):

  def _parse_and_analyze(self, test_fn):
    node, source = parser.parse_entity(test_fn, future_features=())
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file=None,
        future_features=(),
        namespace={})
    node = qual_names.resolve(node)
    namer = naming.Namer({})
    ctx = transformer.Context(entity_info, namer, None)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    graph = graphs[node]
    node = type_inference.resolve(node, graph)
    return node

  def assertTypeInference(self, node, reads, writes):
    type_reads = anno.getanno(node, "type_anno_read")
    type_writes = anno.getanno(node, "type_anno_write")
    self.assertEqual(type_reads, reads)
    self.assertEqual(type_writes, writes)

class TypeInferenceAnalyzerTest(TypeInferenceAnalyzerTestBase):

  def test_type_inference_if_else(self):

    def test_fn(x: ops.Tensor, y: int):
      if x > 0:
        x = x * x + y
      else:
        x = -x
      y = x
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    reads = {x: {tensor_type}}
    writes = {}
    # if x > 0:
    self.assertTypeInference(fn_body[0].test, reads, writes)

    writes = {y: {tensor_type}}
    # y = x
    self.assertTypeInference(fn_body[1], reads, writes)

    writes = {}
    # return x
    self.assertTypeInference(fn_body[2], reads, writes)

  def test_type_inference_if_else_multiple_possible_types(self):

    def test_fn(x: ops.Tensor, y: float):
      if y > 0:
        y = x + x
      return y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    int_type = qual_names.from_str("float")

    reads = {x: {tensor_type}}
    writes = {y: {tensor_type}}
    # y = x + x
    self.assertTypeInference(fn_body[0].body[0], reads, writes)

    reads = {y: {tensor_type, int_type}}
    writes = {}
    # return y
    self.assertTypeInference(fn_body[1], reads, writes)

  def test_type_inference_stacked_if(self):

    def test_fn(x: ops.Tensor, y: int):
      if y > 0:
        x = 0
      if y > 1:
        x = 1
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    reads = {x: {tensor_type}}
    writes = {}
    # return x
    self.assertTypeInference(fn_body[2], reads, writes)

  def test_type_inference_stacked_if_else(self):

    def test_fn(x: ops.Tensor, y: int):
      if y > 0:
        x = 0
      if y > 1:
        x = 1
      else:
        x = 2
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    # reads and writes dicts are empty because type inference for
    # variables assigned to python objects is not supported
    reads = {}
    writes = {}
    # return x
    self.assertTypeInference(fn_body[2], reads, writes)

  def test_type_inference_for(self):

    def test_fn(x: ops.Tensor, y: int):
      for i in range(a):
        x += i
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    reads = {x: {tensor_type}}
    writes = {}

    # return x
    self.assertTypeInference(fn_body[1], reads, writes)


if __name__ == '__main__':
  test.main()
