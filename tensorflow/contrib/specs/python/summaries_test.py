# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for specs-related summarization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.specs.python import specs
from tensorflow.contrib.specs.python import summaries
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class SummariesTest(test.TestCase):

  def testStructure(self):
    with self.test_session():
      inputs_shape = (1, 18, 19, 5)
      inputs = constant_op.constant(_rand(*inputs_shape))
      spec = "net = Cr(64, [5, 5])"
      outputs = specs.create_net(spec, inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 18, 19, 64))
      self.assertEqual(
          summaries.tf_spec_structure(
              spec, input_shape=inputs_shape),
          "_ variablev2 conv variablev2 biasadd relu")

  def testStructureFromTensor(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 18, 19, 5))
      spec = "net = Cr(64, [5, 5])"
      outputs = specs.create_net(spec, inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 18, 19, 64))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 conv variablev2 biasadd relu")

  def testPrint(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 18, 19, 5))
      spec = "net = Cr(64, [5, 5])"
      outputs = specs.create_net(spec, inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 18, 19, 64))
      summaries.tf_spec_print(spec, inputs)

  def testSummary(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 18, 19, 5))
      spec = "net = Cr(64, [5, 5])"
      outputs = specs.create_net(spec, inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 18, 19, 64))
      summaries.tf_spec_summary(spec, inputs)


if __name__ == "__main__":
  test.main()
