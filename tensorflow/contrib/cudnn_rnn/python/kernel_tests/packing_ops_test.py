# -*- coding: utf-8 -*-
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
"""Tests for Cudnn RNN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import unittest

import numpy as np

from tensorflow.contrib.cudnn_rnn.python.ops import packing_ops
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
import tensorflow as tf


class PackingOpsTest(TensorFlowTestCase):


  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testPackedSequenceAlignment(self):
    with ops.Graph().as_default():
	  sequence_lengths = tf.constant([6,3,2], dtype=tf.float32)
	  alignments, batch_sizes = packing_ops.packed_sequence_alignment(sequence_lengths);
	  with self.test_session(
        use_gpu=True, graph=ops.get_default_graph()) as sess:
        _alignments, _batch_sizes = sess.run([alignments, batch_sizes])
        self.assertAllEqual(_alignments, np.arrray([0,3,6,8,9,10]))
        self.assertAllEqual(_batch_sizes, np.arrray([3,3,2,1,1,1]))

if __name__ == "__main__":
  googletest.main()
