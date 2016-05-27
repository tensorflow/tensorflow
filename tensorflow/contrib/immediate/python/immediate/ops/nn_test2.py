# Copyright 2015 Google Inc. All Rights Reserved.
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

# NOTE(yaroslavvb): port of nn_test for immediate execution. The following
# tests are incompatible with immediate execution and are commented out
# 1. Gradient tests (tf.test.compute_gradient_error, tf.gradients)
# 2. Tests that rely on static shape inference (get_shape)


"""Tests for tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import gen_nn_ops

exp = math.exp
log = math.log


from tensorflow.contrib.immediate.python.immediate import test_util
import tensorflow.contrib.immediate as immediate


#env = immediate.Env(tf)
#tf = env.tf
#env2 = immediate.Env(gen_nn_ops)
#gen_nn_ops = env2.tf

env = immediate.Env({"tf": tf, "gen_nn_ops": gen_nn_ops})
tf = env.tf
gen_nn_ops = env.gen_nn_ops


# NOTE(yaroslavvb): SampledLogits don't work because
# embedding ops use generators, and my module rewriter doesn't follow
# references in generators
#   File "/Users/yaroslavvb/tfimmediate_src/tensorflow/bazel-bin/tensorflow/contrib/immediate/nn_test2.runfiles/tensorflow/python/ops/embedding_ops.py", line 84, in embedding_lookup
#    with ops.colocate_with(params[0]):


# class ComputeSampledLogitsTest(test_util.TensorFlowTestCase):

#   def setUp(self):
#     self._num_classes = 5
#     self._dim = 10
#     self._batch_size = 3
#     self._num_shards = 3

#   def _GenerateTestInputs(self):
#     np.random.seed(0)
#     weights = np.random.randn(self._num_classes, self._dim).astype(np.float32)
#     biases = np.random.randn(self._num_classes).astype(np.float32)
#     hidden_acts = np.random.randn(self._batch_size, self._dim).astype(
#         np.float32)
#     sharded_weights = [
#         weights[[row for row in range(self._num_classes)
#                  if row % self._num_shards == shard]]
#         for shard in range(self._num_shards)]
#     return weights, biases, hidden_acts, sharded_weights

#   def _ComputeSampledLogitsNP(self, true_w, true_b, sampled_w, sampled_b,
#                               hidden_acts,
#                               num_true=1,
#                               true_expected=None,
#                               sampled_expected=None):

#     batch_size, dim = hidden_acts.shape
#     true_logits = np.sum(
#         hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape(
#             (batch_size, num_true, dim)),
#         axis=2)
#     true_b = true_b.reshape((batch_size, num_true))
#     true_logits += true_b
#     sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

#     if true_expected is not None:
#       true_logits -= np.log(true_expected)
#     if sampled_expected is not None:
#       sampled_logits -= np.log(sampled_expected[np.newaxis, :])

#     out_logits = np.concatenate([true_logits, sampled_logits], axis=1)
#     out_labels = np.hstack((np.ones_like(true_logits) / num_true,
#                             np.zeros_like(sampled_logits)))

#     return out_logits, out_labels

#   def _ComputeSampledLogitsTF(self, weights, biases, hidden_acts, labels,
#                               num_sampled, num_classes, num_true, sampled_vals,
#                               subtract_log_q, remove_accidental_hits,
#                               name="sampled_loss_TF"):
#     # Should be called from within a `with test_session():` block
#     if isinstance(weights, list):
#       weights_tf = [tf.constant(shard) for shard in weights]
#     else:
#       weights_tf = tf.constant(weights)
#     biases_tf = tf.constant(biases)
#     hidden_acts_tf = tf.constant(hidden_acts,
#                                  shape=(self._batch_size, self._dim))
#     labels_tf = tf.constant(labels,
#                             dtype=tf.int64,
#                             shape=(self._batch_size, num_true))

#     pred_logits_tf, pred_labels_tf = tf.nn._compute_sampled_logits(
#         weights_tf,
#         biases_tf,
#         hidden_acts_tf,
#         labels_tf,
#         num_sampled,
#         num_classes,
#         num_true,
#         sampled_vals,
#         subtract_log_q=subtract_log_q,
#         remove_accidental_hits=remove_accidental_hits,
#         name=name)
#     return pred_logits_tf, pred_labels_tf

#   def testComputeSampledLogitsShapes(self):
#     # We just check that the shapes of the returned values are correct.
#     weights, biases, hidden_acts, _ = self._GenerateTestInputs()
#     sampled = [1, 0, 2, 3]
#     num_sampled = len(sampled)
#     true_exp = sampled_exp = [1., 1., 1., 1.]
#     test_sampled_vals = (sampled, true_exp, sampled_exp)
#     sampled_w, sampled_b = weights[sampled], biases[sampled]

#     with self.test_session() as sess:
#       for num_true_test in range(1, 5):
#         labels = np.random.randint(low=0, high=self._num_classes,
#                                    size=self._batch_size * num_true_test)
#         true_w, true_b = weights[labels], biases[labels]

#         logits_np, labels_np = self._ComputeSampledLogitsNP(
#             true_w, true_b, sampled_w, sampled_b, hidden_acts,
#             num_true=num_true_test)

#         logits_tf, labels_tf = self._ComputeSampledLogitsTF(
#             weights, biases, hidden_acts, labels, num_sampled,
#             self._num_classes,
#             num_true=num_true_test,
#             sampled_vals=test_sampled_vals,
#             remove_accidental_hits=True,
#             subtract_log_q=False)

#       logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
#       self.assertEqual(logits_np.shape, logits_tf_val.shape)
#       self.assertEqual(labels_np.shape, labels_tf_val.shape)

#   def testComputeSampledLogitsValues(self):
#     # Here we check the actual numerics.
#     weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
#     eps = 1e-3
#     sampled = [1, 0, 2, 3]
#     num_sampled = len(sampled)
#     true_exp = np.empty([self._batch_size, 1], dtype=np.float32)
#     true_exp.fill(0.5)
#     sampled_exp = np.empty([num_sampled], dtype=np.float32)
#     sampled_exp.fill(0.5)
#     sampled_w, sampled_b = weights[sampled], biases[sampled]
#     test_sampled_vals = (sampled, true_exp, sampled_exp)

#     with self.test_session() as sess:
#       for num_true_test in range(1, 5):
#         # Generate test data for this run
#         labels = np.random.randint(low=0, high=self._num_classes,
#                                    size=self._batch_size * num_true_test)
#         true_w, true_b = weights[labels], biases[labels]

#         # Test 1: Without accidental hit removal or subtract_log_q
#         logits_np, labels_np = self._ComputeSampledLogitsNP(
#             true_w, true_b, sampled_w, sampled_b, hidden_acts,
#             num_true=num_true_test)
#         logits_tf, labels_tf = self._ComputeSampledLogitsTF(
#             weights, biases, hidden_acts, labels, num_sampled,
#             self._num_classes,
#             num_true=num_true_test,
#             sampled_vals=test_sampled_vals,
#             subtract_log_q=False,
#             remove_accidental_hits=False,
#             name="sampled_loss_test1_num_true%d" % num_true_test)

#         logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
#         self.assertAllClose(logits_np, logits_tf_val, eps)
#         self.assertAllClose(labels_np, labels_tf_val, eps)

#         # Test 2: With accidental hit removal, no subtract_log_q
#         logits_tf, labels_tf = self._ComputeSampledLogitsTF(
#             weights, biases, hidden_acts, labels, num_sampled,
#             self._num_classes,
#             num_true=num_true_test,
#             sampled_vals=test_sampled_vals,
#             subtract_log_q=False,
#             remove_accidental_hits=True,
#             name="sampled_loss_test2_num_true%d" % num_true_test)

#         # Test that the exponentiated logits of accidental hits are near 0.
#         # First we need to find the hits in this random test run:
#         labels_reshape = labels.reshape((self._batch_size, num_true_test))
#         logits_tf_np = logits_tf.eval()
#         for row in xrange(self._batch_size):
#           row_labels = labels_reshape[row, :]
#           for col in xrange(num_sampled):
#             if sampled[col] in row_labels:
#               # We need to add the num_true_test offset into logits_*
#               self.assertNear(
#                   np.exp(logits_tf_np[row, col + num_true_test]), 0., eps)

#         # Test 3: With subtract_log_q, no accidental hit removal
#         logits_np, labels_np = self._ComputeSampledLogitsNP(
#             true_w, true_b, sampled_w, sampled_b, hidden_acts,
#             num_true=num_true_test,
#             true_expected=true_exp,
#             sampled_expected=sampled_exp)
#         logits_tf, labels_tf = self._ComputeSampledLogitsTF(
#             weights, biases, hidden_acts, labels, num_sampled,
#             self._num_classes,
#             num_true=num_true_test,
#             sampled_vals=test_sampled_vals,
#             subtract_log_q=True,
#             remove_accidental_hits=False,
#             name="sampled_loss_test3_num_true%d" % num_true_test)

#         logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
#         self.assertAllClose(logits_np, logits_tf_val, eps)
#         self.assertAllClose(labels_np, labels_tf_val, eps)

#         # Test 4: Test 1, with sharded weights
#         logits_np, labels_np = self._ComputeSampledLogitsNP(
#             true_w, true_b, sampled_w, sampled_b, hidden_acts,
#             num_true=num_true_test)
#         logits_tf, labels_tf = self._ComputeSampledLogitsTF(
#             sharded_weights, biases, hidden_acts, labels, num_sampled,
#             self._num_classes,
#             num_true=num_true_test,
#             sampled_vals=test_sampled_vals,
#             subtract_log_q=False,
#             remove_accidental_hits=False,
#             name="sampled_loss_test1_num_true%d" % num_true_test)

#         logits_tf_val, labels_tf_val = sess.run([logits_tf, labels_tf])
#         self.assertAllClose(logits_np, logits_tf_val, eps)
#         self.assertAllClose(labels_np, labels_tf_val, eps)

#   def testNCELoss(self):
#     # A simple test to verify the numerics.

#     def _SigmoidCrossEntropyWithLogits(logits, targets):
#       # logits, targets: float arrays of the same shape.
#       assert logits.shape == targets.shape
#       pred = 1. / (1. + np.exp(-logits))
#       eps = 0.0001
#       pred = np.minimum(np.maximum(pred, eps), 1 - eps)
#       return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

#     weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
#     labels = [0, 1, 2]
#     true_w, true_b = weights[labels], biases[labels]
#     sampled = [1, 0, 2, 3]
#     num_sampled = len(sampled)
#     true_exp = np.empty([self._batch_size, 1], dtype=np.float32)
#     true_exp.fill(0.5)
#     sampled_exp = np.empty([num_sampled], dtype=np.float32)
#     sampled_exp.fill(0.5)
#     sampled_w, sampled_b = weights[sampled], biases[sampled]
#     test_sampled_vals = (sampled, true_exp, sampled_exp)

#     with self.test_session():
#       logits_np, labels_np = self._ComputeSampledLogitsNP(
#           true_w, true_b, sampled_w, sampled_b, hidden_acts,
#           true_expected=true_exp,
#           sampled_expected=sampled_exp)
#       nce_loss_np = np.sum(
#           _SigmoidCrossEntropyWithLogits(logits_np, labels_np), 1)

#       labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
#       weights_tf = tf.constant(weights)
#       biases_tf = tf.constant(biases)
#       inputs_tf = tf.constant(hidden_acts)

#       nce_loss_tf = tf.nn.nce_loss(weights_tf,
#                                    biases_tf,
#                                    inputs_tf,
#                                    labels_tf,
#                                    num_sampled=1,
#                                    num_classes=self._num_classes,
#                                    num_true=1,
#                                    sampled_values=test_sampled_vals)

#       self.assertAllClose(nce_loss_np, nce_loss_tf.eval(), 1e-4)

#       # Test with sharded weights
#       nce_loss_tf = tf.nn.nce_loss(
#           [tf.constant(shard) for shard in sharded_weights],
#           biases_tf,
#           inputs_tf,
#           labels_tf,
#           num_sampled=1,
#           num_classes=self._num_classes,
#           num_true=1,
#           sampled_values=test_sampled_vals)

#       self.assertAllClose(nce_loss_np, nce_loss_tf.eval(), 1e-4)

#   def testSampledSoftmaxLoss(self):
#     # A simple test to verify the numerics.

#     def _SoftmaxCrossEntropyWithLogits(logits, targets):
#       # logits, targets: float arrays of the same shape.
#       assert logits.shape == targets.shape
#       stable_exp_logits = np.exp(logits - np.amax(
#           logits, axis=1, keepdims=True))
#       pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
#       return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

#     weights, biases, hidden_acts, sharded_weights = self._GenerateTestInputs()
#     labels = [0, 1, 2]
#     true_w, true_b = weights[labels], biases[labels]
#     sampled = [1, 0, 2, 3]
#     num_sampled = len(sampled)
#     true_exp = np.full([self._batch_size, 1], fill_value=0.5, dtype=np.float32)
#     sampled_exp = np.full([num_sampled], fill_value=0.5, dtype=np.float32)
#     sampled_w, sampled_b = weights[sampled], biases[sampled]
#     test_sampled_vals = (sampled, true_exp, sampled_exp)

#     with self.test_session():
#       logits_np, labels_np = self._ComputeSampledLogitsNP(
#           true_w, true_b, sampled_w, sampled_b, hidden_acts,
#           true_expected=true_exp,
#           sampled_expected=sampled_exp)
#       sampled_softmax_loss_np = _SoftmaxCrossEntropyWithLogits(logits_np,
#                                                                labels_np)

#       labels_tf = tf.constant(labels, shape=(self._batch_size, 1))
#       weights_tf = tf.constant(weights)
#       biases_tf = tf.constant(biases)
#       inputs_tf = tf.constant(hidden_acts)

#       sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
#           weights_tf,
#           biases_tf,
#           inputs_tf,
#           labels_tf,
#           num_sampled=1,
#           num_classes=self._num_classes,
#           num_true=1,
#           sampled_values=test_sampled_vals,
#           remove_accidental_hits=False)

#       self.assertAllClose(
#           sampled_softmax_loss_np, sampled_softmax_loss_tf.eval(), 1e-4)

#       # Test with sharded weights
#       sampled_softmax_loss_tf = tf.nn.sampled_softmax_loss(
#           [tf.constant(shard) for shard in sharded_weights],
#           biases_tf,
#           inputs_tf,
#           labels_tf,
#           num_sampled=1,
#           num_classes=self._num_classes,
#           num_true=1,
#           sampled_values=test_sampled_vals,
#           remove_accidental_hits=False)

#       self.assertAllClose(
#           sampled_softmax_loss_np, sampled_softmax_loss_tf.eval(), 1e-4)


if __name__ == "__main__":
  tf.test.main()
