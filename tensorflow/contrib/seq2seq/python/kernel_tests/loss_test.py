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

"""Tests for contrib.seq2seq.python.seq2seq.loss_ops."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf

class LossTest(tf.test.TestCase):

  def testSequenceLoss(self):
    with self.test_session() as sess:
      with tf.variable_scope("root",
          initializer=tf.constant_initializer(0.5)) as varscope:
        batch_size = 2
        sequence_length = 3
        number_of_classes = 5
        logits = [tf.constant(i + 0.5, shape=[batch_size, number_of_classes])
                  for i in range(sequence_length)]
        logits = tf.stack(logits, axis=1)
        targets = [tf.constant(i, tf.int32, shape=[batch_size]) for i in
                   range(sequence_length)]
        targets = tf.stack(targets, axis=1)
        weights = [tf.constant(1.0, shape=[batch_size]) for i in
                   range(sequence_length)]
        weights = tf.stack(weights, axis=1)

        average_loss_per_example = tf.contrib.seq2seq.sequence_loss(
            logits, targets, weights,
            average_across_timesteps=True,
            average_across_batch=True)
        res = sess.run(average_loss_per_example)
        self.assertAllClose(1.60944, res)

        average_loss_per_sequence = tf.contrib.seq2seq.sequence_loss(
            logits, targets, weights,
            average_across_timesteps=False,
            average_across_batch=True)
        res = sess.run(average_loss_per_sequence)
        compare_per_sequence = np.ones((sequence_length)) * 1.60944
        self.assertAllClose(compare_per_sequence, res)

        average_loss_per_batch = tf.contrib.seq2seq.sequence_loss(
            logits, targets, weights,
            average_across_timesteps=True,
            average_across_batch=False)
        res = sess.run(average_loss_per_batch)
        compare_per_batch = np.ones((batch_size)) * 1.60944
        self.assertAllClose(compare_per_batch, res)

        total_loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, weights,
            average_across_timesteps=False,
            average_across_batch=False)
        res = sess.run(total_loss)
        compare_total = np.ones((batch_size, sequence_length)) * 1.60944
        self.assertAllClose(compare_total, res)

if __name__ == '__main__':
  tf.test.main()
