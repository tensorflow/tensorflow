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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import nn
from tensorflow.contrib.labeled_tensor.python.ops import test_util
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops


class NNTests(test_util.Base):

  def setUp(self):
    super(NNTests, self).setUp()
    self.axes = ['x']
    self.original_lt = core.LabeledTensor([0.0, 0.5, 1.0], self.axes)
    self.other_lt = 1 - self.original_lt

  def test_unary_ops(self):
    ops = [
        ('relu', nn_ops.relu, nn.relu),
        ('relu6', nn_ops.relu6, nn.relu6),
        ('crelu', nn_ops.crelu, nn.crelu),
        ('elu', nn_ops.elu, nn.elu),
        ('softplus', nn_ops.softplus, nn.softplus),
        ('l2_loss', nn_ops.l2_loss, nn.l2_loss),
        ('softmax', nn_ops.softmax, nn.softmax),
        ('log_softmax', nn_ops.log_softmax, nn.log_softmax),
    ]
    for op_name, tf_op, lt_op in ops:
      golden_tensor = tf_op(self.original_lt.tensor)
      golden_lt = core.LabeledTensor(golden_tensor, self.axes)
      actual_lt = lt_op(self.original_lt)
      self.assertIn(op_name, actual_lt.name)
      self.assertLabeledTensorsEqual(golden_lt, actual_lt)

  def test_binary_ops(self):
    ops = [
        ('sigmoid_cross_entropy_with_logits',
         nn_impl.sigmoid_cross_entropy_with_logits,
         nn.sigmoid_cross_entropy_with_logits),
        ('softmax_cross_entropy_with_logits',
         nn_ops.softmax_cross_entropy_with_logits,
         nn.softmax_cross_entropy_with_logits),
        ('sparse_softmax_cross_entropy_with_logits',
         nn_ops.sparse_softmax_cross_entropy_with_logits,
         nn.sparse_softmax_cross_entropy_with_logits),
    ]
    for op_name, tf_op, lt_op in ops:
      golden_tensor = tf_op(self.original_lt.tensor, self.other_lt.tensor)
      golden_lt = core.LabeledTensor(golden_tensor, self.axes)
      actual_lt = lt_op(self.original_lt, self.other_lt)
      self.assertIn(op_name, actual_lt.name)
      self.assertLabeledTensorsEqual(golden_lt, actual_lt)
