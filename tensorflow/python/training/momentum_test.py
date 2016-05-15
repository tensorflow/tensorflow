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

"""Tests for Momentum."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class MomentumOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        mom_opt = tf.train.MomentumOptimizer(learning_rate=2.0, momentum=0.9)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Check we have slots
        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        self.assertFalse(slot0 in tf.trainable_variables())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        self.assertFalse(slot1 in tf.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), slot0.eval())
        self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(np.array([1.0 - (0.1 * 2.0),
                                                     2.0 - (0.1 * 2.0)]),
                                           var0.eval())
        self.assertAllCloseAccordingToType(np.array([3.0 - (0.01 * 2.0),
                                                     4.0 - (0.01 * 2.0)]),
                                           var1.eval())
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]),
            slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
            slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                      2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                      3.98 - ((0.9 * 0.01 + 0.01) * 2.0)]),
            var1.eval())

  def testTensorLearningRateAndMomentum(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        mom_opt = tf.train.MomentumOptimizer(
            learning_rate=tf.constant(2.0), momentum=tf.constant(0.9))
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Check we have slots
        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        self.assertFalse(slot0 in tf.trainable_variables())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        self.assertFalse(slot1 in tf.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), slot0.eval())
        self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(np.array([1.0 - (0.1 * 2.0),
                                                     2.0 - (0.1 * 2.0)]),
                                           var0.eval())
        self.assertAllCloseAccordingToType(np.array([3.0 - (0.01 * 2.0),
                                                     4.0 - (0.01 * 2.0)]),
                                           var1.eval())
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]),
            slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
            slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                      2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                      3.98 - ((0.9 * 0.01 + 0.01) * 2.0)]),
            var1.eval())

  def testFloat64(self):
    with self.test_session():
      opt = tf.train.MomentumOptimizer(learning_rate=2.0, momentum=0.9)

      # compute_gradients.
      values = [1.0, 3.0]
      good_vars = [tf.Variable([v]) for v in values]
      bad_loss = tf.constant(2.0, tf.float64, name="bad_loss")
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_loss.*expected.*float32",
          opt.compute_gradients, bad_loss, good_vars)
      bad_vars = [
          tf.Variable(np.array([v], np.float64), name="bad_var")
          for v in values]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.compute_gradients, tf.cast(bad_vars[0] + bad_vars[1], tf.float32),
          bad_vars)
      opt.compute_gradients(good_vars[0] + good_vars[1], good_vars)

      # apply_gradients.
      bad_grads = [
          tf.constant([0.1], dtype=np.float64, name="bad_grad"),
          tf.constant([0.01])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_grad.*expected.*float32",
          opt.apply_gradients, zip(bad_grads, good_vars))
      good_grads = [tf.constant([0.01]), tf.constant([0.02])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.apply_gradients, zip(good_grads, bad_vars))
      opt.apply_gradients(zip(good_grads, good_vars))

  def _dbParamsMom01(self):
    """Return dist-belief momentum values.

    Return values been generated from the dist-belief momentum unittest,
    running with a learning rate of 0.1 and a momentum of 0.1.

    These values record how a parameter vector of size 10, initialized with 0.0,
    gets updated with 10 consecutive momentum steps.  It uses random gradients.

    Returns:
      db_grad: The gradients to apply
      db_out: The parameters after the momentum update.
    """
    db_grad = [[]] * 10
    db_out = [[]] * 10
    # pylint: disable=line-too-long
    db_grad[0] = [0.00096264342, 0.17914793, 0.93945462, 0.41396621, 0.53037018, 0.93197989, 0.78648776, 0.50036013, 0.55345792, 0.96722615]
    db_out[0] = [-9.6264346e-05, -0.017914793, -0.093945466, -0.041396622, -0.053037018, -0.093197994, -0.078648776, -0.050036013, -0.055345792, -0.096722618]
    db_grad[1] = [0.17075552, 0.88821375, 0.20873757, 0.25236958, 0.57578111, 0.15312378, 0.5513742, 0.94687688, 0.16012503, 0.22159521]
    db_out[1] = [-0.017181443, -0.10852765, -0.12421377, -0.070773244, -0.11591884, -0.11783017, -0.14165108, -0.14972731, -0.076892875, -0.1285544]
    db_grad[2] = [0.35077485, 0.47304362, 0.44412705, 0.44368884, 0.078527533, 0.81223965, 0.31168157, 0.43203235, 0.16792089, 0.24644311]
    db_out[2] = [-0.053967446, -0.1648933, -0.1716533, -0.1180798, -0.13005978, -0.20151734, -0.17911947, -0.20289968, -0.095839672, -0.15638189]
    db_grad[3] = [0.9694621, 0.75035888, 0.28171822, 0.83813518, 0.53807181, 0.3728098, 0.81454384, 0.03848977, 0.89759839, 0.93665648]
    db_out[3] = [-0.15459226, -0.24556576, -0.20456907, -0.20662397, -0.18528105, -0.24716705, -0.2643207, -0.21206589, -0.18749419, -0.2528303]
    db_grad[4] = [0.38578293, 0.8536852, 0.88722926, 0.66276771, 0.13678469, 0.94036359, 0.69107032, 0.81897682, 0.5433259, 0.67860287]
    db_out[4] = [-0.20323303, -0.33900154, -0.29658359, -0.28175515, -0.20448165, -0.34576839, -0.34194785, -0.29488021, -0.25099224, -0.33033544]
    db_grad[5] = [0.27885768, 0.76100707, 0.24625534, 0.81354135, 0.18959245, 0.48038563, 0.84163809, 0.41172323, 0.83259648, 0.44941229]
    db_out[5] = [-0.23598288, -0.42444581, -0.33041057, -0.3706224, -0.22536094, -0.40366709, -0.43387437, -0.34433398, -0.34060168, -0.38302717]
    db_grad[6] = [0.27233034, 0.056316052, 0.5039115, 0.24105175, 0.35697976, 0.75913221, 0.73577434, 0.16014607, 0.57500273, 0.071136251]
    db_out[6] = [-0.26649091, -0.43862185, -0.38418442, -0.40361428, -0.26314685, -0.48537019, -0.51664448, -0.36529395, -0.40706289, -0.39540997]
    db_grad[7] = [0.58697265, 0.2494842, 0.08106143, 0.39954534, 0.15892942, 0.12683646, 0.74053431, 0.16033, 0.66625422, 0.73515922]
    db_out[7] = [-0.32823896, -0.46498787, -0.39766794, -0.446868, -0.28281838, -0.50622416, -0.59897494, -0.38342294, -0.48033443, -0.47016418]
    db_grad[8] = [0.8215279, 0.41994119, 0.95172721, 0.68000203, 0.79439718, 0.43384039, 0.55561525, 0.22567581, 0.93331909, 0.29438227]
    db_out[8] = [-0.41656655, -0.50961858, -0.49418902, -0.51919359, -0.36422527, -0.55169362, -0.6627695, -0.40780342, -0.58099347, -0.50707781]
    db_grad[9] = [0.68297005, 0.67758518, 0.1748755, 0.13266537, 0.70697063, 0.055731893, 0.68593478, 0.50580865, 0.12602448, 0.093537711]
    db_out[9] = [-0.49369633, -0.58184016, -0.52132869, -0.5396927, -0.44306302, -0.56181377, -0.73774242, -0.46082234, -0.60366184, -0.52012295]
    # pylint: enable=line-too-long
    return db_grad, db_out

  def testLikeDistBeliefMom01(self):
    with self.test_session():
      db_grad, db_out = self._dbParamsMom01()
      num_samples = len(db_grad)
      var0 = tf.Variable([0.0] * num_samples)
      grads0 = tf.constant([0.0] * num_samples)
      mom_opt = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.1)
      mom_update = mom_opt.apply_gradients(zip([grads0], [var0]))
      tf.initialize_all_variables().run()
      for i in xrange(num_samples):
        mom_update.run(feed_dict={grads0: db_grad[i]})
        self.assertAllClose(np.array(db_out[i]), var0.eval())

  def testSparse(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable(tf.zeros([4, 2], dtype=dtype))
        var1 = tf.Variable(tf.constant(1.0, dtype, [4, 2]))
        grads0 = tf.IndexedSlices(tf.constant([[.1, .1]], dtype=dtype),
                                  tf.constant([1]),
                                  tf.constant([4, 2]))
        grads1 = tf.IndexedSlices(tf.constant([[.01, .01], [.01, .01]],
                                              dtype=dtype),
                                  tf.constant([2, 3]),
                                  tf.constant([4, 2]))
        mom_opt = tf.train.MomentumOptimizer(learning_rate=2.0, momentum=0.9)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()

        # Check we have slots
        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([0, 0], var0.eval()[0])
        self.assertAllClose([0, 0], var0.eval()[1])
        self.assertAllClose([1, 1], var1.eval()[2])

        # Step 1: the momentum accumulators are 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([0, 0]), slot0.eval()[0])
        self.assertAllCloseAccordingToType(
            np.array([.1, .1]), slot0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([.01, .01]), slot1.eval()[2])
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(np.array([0, 0]), var0.eval()[0])
        self.assertAllCloseAccordingToType(np.array([- (0.1 * 2.0),
                                                     - (0.1 * 2.0)]),
                                           var0.eval()[1])
        self.assertAllCloseAccordingToType(np.array([1.0 - (0.01 * 2.0),
                                                     1.0 - (0.01 * 2.0)]),
                                           var1.eval()[2])
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllClose(np.array([0, 0]), slot0.eval()[0])
        self.assertAllCloseAccordingToType(np.array([(0.9 * 0.1 + 0.1),
                                                     (0.9 * 0.1 + 0.1)]),
                                           slot0.eval()[1])
        self.assertAllCloseAccordingToType(np.array([(0.9 * 0.01 + 0.01),
                                                     (0.9 * 0.01 + 0.01)]),
                                           slot1.eval()[2])
        # Check that the parameters have been updated.
        self.assertAllClose(np.array([0, 0]), var0.eval()[0])
        self.assertAllCloseAccordingToType(
            np.array([- (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                      - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)]),
            var0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([0.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                      0.98 - ((0.9 * 0.01 + 0.01) * 2.0)]),
            var1.eval()[2])

  def testSharing(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        mom_opt = tf.train.MomentumOptimizer(learning_rate=2.0, momentum=0.9)
        mom_update1 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        mom_update2 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()

        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update1.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), slot0.eval())
        self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(np.array([1.0 - (0.1 * 2.0),
                                                     2.0 - (0.1 * 2.0)]),
                                           var0.eval())
        self.assertAllCloseAccordingToType(np.array([3.0 - (0.01 * 2.0),
                                                     4.0 - (0.01 * 2.0)]),
                                           var1.eval())
        # Step 2: the second momentum accumulators contain the previous update.
        mom_update2.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]),
            slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
            slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                      2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
                      3.98 - ((0.9 * 0.01 + 0.01) * 2.0)]),
            var1.eval())


if __name__ == "__main__":
  tf.test.main()
