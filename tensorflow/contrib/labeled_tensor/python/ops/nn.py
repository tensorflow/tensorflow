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

"""Neural network ops for LabeledTensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.python.ops import nn


relu = core.define_unary_op('relu', nn.relu)
relu6 = core.define_unary_op('relu6', nn.relu6)
crelu = core.define_unary_op('crelu', nn.crelu)
elu = core.define_unary_op('elu', nn.elu)
softplus = core.define_unary_op('softplus', nn.softplus)

l2_loss = core.define_unary_op('l2_loss', nn.l2_loss)
sigmoid_cross_entropy_with_logits = core.define_binary_op(
    'sigmoid_cross_entropy_with_logits',
    nn.sigmoid_cross_entropy_with_logits)
softmax = core.define_unary_op('softmax', nn.softmax)
log_softmax = core.define_unary_op('log_softmax', nn.log_softmax)
softmax_cross_entropy_with_logits = core.define_binary_op(
    'softmax_cross_entropy_with_logits',
    nn.softmax_cross_entropy_with_logits)
sparse_softmax_cross_entropy_with_logits = core.define_binary_op(
    'sparse_softmax_cross_entropy_with_logits',
    nn.sparse_softmax_cross_entropy_with_logits)
