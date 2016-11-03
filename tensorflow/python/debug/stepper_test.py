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
"""Unit tests of the tfdbg Stepper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.debug.stepper import NodeStepper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class StepperTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.a = tf.Variable(2.0, name="a")
    self.b = tf.Variable(3.0, name="b")

    self.c = tf.mul(self.a, self.b, name="c")  # Should be 6.0.
    self.d = tf.mul(self.a, self.a, name="d")  # Should be 4.0.

    self.e = tf.mul(self.d, self.c, name="e")  # Should be 24.0.

    self.f = tf.div(self.b, 0.30, name="f")  # Should be 20.0.

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def tearDown(self):
    tf.reset_default_graph()

  def testAttemptToContToFetchNotInTransitiveClosure(self):
    stepper = NodeStepper(self.sess, "e:0")

    self.assertEqual(
        ["a:0", "b:0", "b/read:0", "a/read:0", "c:0", "d:0", "e:0"],
        stepper.sorted_transitive_closure())

    with self.assertRaisesRegexp(
        ValueError,
        "Target \"f:0\" is not in the transitive closure for the fetch of the "
        "stepper: \"e:0\""):
      stepper.cont("f:0")

  def testUsingNamesNotUsingIntermediateTensors(self):
    stepper = NodeStepper(self.sess, "e:0")

    # The first cont() call should have used no feeds.
    result = stepper.cont("c:0")
    self.assertAllClose(6.0, result)
    self.assertEqual({}, stepper.last_feed_types())

    # The second cont() call should have used the tensor handle from the
    # previous cont() call.
    result = stepper.cont("e:0")
    self.assertAllClose(24.0, result)
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

  def testUsingNodesNotUsingIntermediateTensors(self):
    stepper = NodeStepper(self.sess, self.e)

    # There should be no handles before any cont() calls.
    self.assertEqual([], stepper.handle_names())

    # Before the cont() call, the stepper should not have access to the value
    # of c:0.
    with self.assertRaisesRegexp(
        ValueError,
        "This stepper instance does not have access to the value of tensor "
        "\"c:0\""):
      stepper.get_tensor_value("c:0")

    # Using the node/tensor itself, instead of the name str, should work on
    # cont().
    result = stepper.cont(self.c)
    self.assertAllClose(6.0, result)
    self.assertEqual({}, stepper.last_feed_types())

    self.assertEqual(["c:0"], stepper.handle_names())

    # After the cont() call, the stepper should have access to the value of c:0
    # via a tensor handle.
    self.assertAllClose(6.0, stepper.get_tensor_value("c:0"))

    result = stepper.cont(self.e)
    self.assertAllClose(24.0, result)
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

  def testIsFeedable(self):
    stepper = NodeStepper(self.sess, self.e)

    self.assertTrue(stepper.is_feedable("a/read:0"))
    self.assertTrue(stepper.is_feedable("b/read:0"))
    self.assertTrue(stepper.is_feedable("c:0"))
    self.assertTrue(stepper.is_feedable("d:0"))

  def testOverrideValue(self):
    stepper = NodeStepper(self.sess, self.e)

    result = stepper.cont(self.c)
    self.assertAllClose(6.0, result)
    self.assertEqual({}, stepper.last_feed_types())

    # There should be no overrides before any cont() calls.
    self.assertEqual([], stepper.override_names())

    # Calling cont() on c again should lead to use of the handle.
    result = stepper.cont(self.c)
    self.assertAllClose(6.0, result)
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # Override c:0.
    stepper.override_tensor("c:0", 7.0)

    # After the overriding, calling get_tensor_value() on c:0 should yield the
    # overriding value.
    self.assertEqual(7.0, stepper.get_tensor_value("c:0"))

    # Now c:0 should have only an override value, but no cached handle, because
    # the handle should have been invalidated.
    self.assertEqual([], stepper.handle_names())
    self.assertEqual(["c:0"], stepper.override_names())

    # Run a downstream tensor after the value override.
    result = stepper.cont(self.e)
    self.assertAllClose(28.0, result)  # Should reflect the overriding value.

    # Should use override, instead of the handle.
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

  def testOverrideValueTwice(self):
    stepper = NodeStepper(self.sess, self.e)

    # Override once.
    stepper.override_tensor("c:0", 7.0)
    self.assertAllClose(28.0, stepper.cont(self.e))
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

    self.assertEqual(["e:0"], stepper.handle_names())
    self.assertEqual(["c:0"], stepper.override_names())

    # Calling cont(self.e) again. This time the cached tensor handle of e
    # should be used.
    self.assertEqual(28.0, stepper.cont(self.e))
    self.assertEqual({
        "e:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # Override c again. This should have invalidated the cache for e.
    stepper.override_tensor("c:0", 8.0)

    self.assertEqual([], stepper.handle_names())
    self.assertEqual(["c:0"], stepper.override_names())

    self.assertAllClose(32.0, stepper.cont(self.e))
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

  def testRemoveOverrideValue(self):
    stepper = NodeStepper(self.sess, self.e)

    result = stepper.cont(self.c)
    self.assertAllClose(6.0, result)
    self.assertEqual({}, stepper.last_feed_types())

    # The previous cont() step should have generated a cached tensor handle.
    self.assertEqual(["c:0"], stepper.handle_names())

    # Override c:0.
    stepper.override_tensor("c:0", 7.0)

    # The overriding should have invalidated the tensor handle.
    self.assertEqual([], stepper.handle_names())
    self.assertEqual(["c:0"], stepper.override_names())

    result = stepper.cont(self.e)
    self.assertAllClose(28.0, result)  # Should reflect the overriding value.
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

    # The handle to tensor e:0 should have been cached, even though its
    # transitive closure contains an override.
    self.assertIn("e:0", stepper.handle_names())

    # Remove the override.
    stepper.remove_override("c:0")
    # c:0 should not be in the overrides anymore.
    self.assertEqual([], stepper.override_names())

    # Removing the override should have invalidated the tensor handle for c.
    self.assertNotIn("e:0", stepper.handle_names())

    # Should reflect the non-overriding value.
    self.assertAllClose(24.0, stepper.cont(self.e))

    # This time, the handle to tensor e:0 should have been cached again, even
    # thought its transitive closure contains an override.
    self.assertIn("e:0", stepper.handle_names())

    # Calling cont(self.e) again should have used the tensor handle to e:0.
    self.assertAllClose(24.0, stepper.cont(self.e))
    self.assertEqual({
        "e:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

  def testOverrideAndContToSameTensor(self):
    stepper = NodeStepper(self.sess, self.e)

    result = stepper.cont(self.c)
    self.assertAllClose(6.0, result)
    self.assertEqual({}, stepper.last_feed_types())
    self.assertEqual(["c:0"], stepper.handle_names())

    self.assertAllClose(6.0, stepper.cont(self.c))

    # The last cont() call should use the tensor handle directly.
    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # Override c:0.
    stepper.override_tensor("c:0", 7.0)

    # As a result of the override, the tensor handle should have been
    # invalidated.
    self.assertEqual([], stepper.handle_names())

    result = stepper.cont(self.c)
    self.assertAllClose(7.0, result)

    self.assertEqual({
        "c:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

  def testFinalizeWithPreviousOverrides(self):
    stepper = NodeStepper(self.sess, self.e)

    stepper.override_tensor("a/read:0", 20.0)
    self.assertEqual(["a/read:0"], stepper.override_names())

    # Should reflect the overriding value.
    self.assertAllClose(24000.0, stepper.cont("e:0"))
    self.assertEqual({
        "a/read:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

    # Finalize call should have ignored the overriding value.
    self.assertAllClose(24.0, stepper.finalize())

  def testRemoveNonexistentOverrideValue(self):
    stepper = NodeStepper(self.sess, self.e)
    self.assertEqual([], stepper.override_names())

    with self.assertRaisesRegexp(
        ValueError, "No overriding value exists for tensor \"c:0\""):
      stepper.remove_override("c:0")

  def testAttemptToOverrideInvalidTensor(self):
    stepper = NodeStepper(self.sess, self.e)

    with self.assertRaisesRegexp(ValueError, "Cannot override tensor \"f:0\""):
      stepper.override_tensor("f:0", 42.0)

  def testInvalidOverrideArgumentType(self):
    stepper = NodeStepper(self.sess, self.e)

    with self.assertRaisesRegexp(TypeError, "Expected type str; got type"):
      stepper.override_tensor(self.a, 42.0)


class StepperTestWithPlaceHolders(test_util.TensorFlowTestCase):

  def setUp(self):
    self.ph0 = tf.placeholder(tf.float32, shape=(2, 2), name="ph0")
    self.ph1 = tf.placeholder(tf.float32, shape=(2, 1), name="ph1")

    self.x = tf.matmul(self.ph0, self.ph1, name="x")
    self.y = tf.add(self.x, self.ph1, name="y")

    self.sess = tf.Session()

  def tearDown(self):
    tf.reset_default_graph()

  def testContWithPlaceholders(self):
    stepper = NodeStepper(
        self.sess,
        self.y,
        feed_dict={
            self.ph0: [[1.0, 2.0], [-3.0, 5.0]],
            self.ph1: [[-1.0], [0.5]]
        })

    self.assertEqual(["ph0:0", "ph1:0", "x:0", "y:0"],
                     stepper.sorted_transitive_closure())

    result = stepper.cont(self.x)
    self.assertAllClose([[0.0], [5.5]], result)
    self.assertEqual({
        "ph0:0": NodeStepper.FEED_TYPE_CLIENT,
        "ph1:0": NodeStepper.FEED_TYPE_CLIENT,
    }, stepper.last_feed_types())

    self.assertEqual(["x:0"], stepper.handle_names())

    result = stepper.cont(self.y)
    self.assertAllClose([[-1.0], [6.0]], result)
    self.assertEqual({
        "x:0": NodeStepper.FEED_TYPE_HANDLE,
        "ph1:0": NodeStepper.FEED_TYPE_CLIENT,
    }, stepper.last_feed_types())

  def testAttemptToContToPlaceholder(self):
    stepper = NodeStepper(
        self.sess,
        self.y,
        feed_dict={
            self.ph0: [[1.0, 2.0], [-3.0, 5.0]],
            self.ph1: [[-1.0], [0.5]]
        })

    with self.assertRaisesRegexp(ValueError,
                                 r"Should not call cont\(\) on a Placeholder"):
      stepper.cont(self.ph0)


class StepperBackwardRunTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Test setup.

    Structure of the forward graph:
              f
             | |
        -----   -----
        |           |
        d           e
       | |         | |
    ---   ---------  ---
    |         |        |
    a         b        c

    Construct a backward graph using the GradientDescentOptimizer.
    """

    self.a = tf.Variable(1.0, name="a")
    self.b = tf.Variable(2.0, name="b")
    self.c = tf.Variable(4.0, name="c")
    self.d = tf.mul(self.a, self.b, name="d")
    self.e = tf.mul(self.b, self.c, name="e")
    self.f = tf.mul(self.d, self.e, name="f")

    # Gradient descent optimizer that minimizes g.
    tf.train.GradientDescentOptimizer(0.01).minimize(self.f, name="optim")

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def tearDown(self):
    tf.reset_default_graph()

  def testContToUpdateA(self):
    stepper = NodeStepper(self.sess, "optim")

    result = stepper.cont("a:0")
    self.assertAllClose(1.0, result)
    self.assertEqual({}, stepper.last_feed_types())

    result = stepper.cont("optim/learning_rate:0")
    self.assertAllClose(0.01, result)
    self.assertEqual({}, stepper.last_feed_types())

    # Before any cont calls on ApplyGradientDescent, there should be no "dirty"
    # variables.
    self.assertEqual(set(), stepper.dirty_variables())

    # First, all the two control inputs to optim.
    result = stepper.cont("optim/update_a/ApplyGradientDescent")

    # Now variable a should have been marked as dirty due to the update
    # by optim/update_a/ApplyGradientDescent.
    self.assertEqual({"a:0"}, stepper.dirty_variables())
    self.assertIsNone(result)
    self.assertEqual({
        "optim/learning_rate:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # Check that Variable "a" has been updated properly, but "b", "c" and "d"
    # remain the same.
    # For backprop on Variable a:
    #   Because f = a * b * b * c, df / da = b * b * c.
    #   1.0 - learning_rate * b * b * c
    #     = 1.0 -  0.01 * 2.0 * 2.0 * 4.0 = 0.84.
    self.assertAllClose(0.84, self.sess.run(self.a))
    self.assertAllClose(2.0, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

  def testContToUpdateB(self):
    stepper = NodeStepper(self.sess, "optim")

    result = stepper.cont("optim/update_b/ApplyGradientDescent")
    self.assertIsNone(result)
    self.assertEqual(set(["b:0"]), stepper.dirty_variables())

    # For backprop on Variable b:
    #   Because f = a * b * b * c, df / da = 2 * a * b * c.
    #   2.0 - learning_rate * 2 * a * b * c
    #     = 2.0 - 0.01 * 2 * 1.0 * 2.0 * 4.0 = 1.84
    self.assertAllClose(1.0, self.sess.run(self.a))
    self.assertAllClose(1.84, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

  def testContAfterUpdateWithoutRestoringVariableValue(self):
    stepper = NodeStepper(self.sess, "optim")

    # First, update Variable a from 1.0 to 0.84.
    result = stepper.cont("optim/update_a/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)
    self.assertEqual(set(["a:0"]), stepper.dirty_variables())
    self.assertAllClose(0.84, self.sess.run(self.a))
    self.assertAllClose(2.0, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

    # Second, update Variable b without the default restore_variable_values.
    result = stepper.cont(
        "optim/update_b/ApplyGradientDescent", restore_variable_values=False)
    self.assertIsNone(result)
    # For the backprop on Variable b under the updated value of a:
    #   2.0 - learning_rate * 2 * a' * b * c
    #     = 2.0 - 0.01 * 2 * 0.84 * 2.0 * 4.0 = 1.8656
    self.assertAllClose(0.84, self.sess.run(self.a))
    self.assertAllClose(1.8656, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

  def testUpdateTwiceRestoreVariable(self):
    stepper = NodeStepper(self.sess, "optim")

    result = stepper.cont("optim/update_a/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)
    self.assertEqual({"a:0"}, stepper.dirty_variables())

    result = stepper.cont("optim/update_b/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)
    # Variables a and c should have been restored and hence no longer dirty.
    # Variable b should have been marked as dirty.
    self.assertEqual({"b:0"}, stepper.dirty_variables())

    # The result of the update should be identitcal to as if only update_b is
    # run.
    self.assertAllClose(1.0, self.sess.run(self.a))
    self.assertAllClose(1.84, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

  def testSelectiveHandleUsageDependingOnTransitiveCleanliness(self):
    """Test tensor handlers are using only during clean transitive closure.

    "clean" means no Variables have been updated by preceding cont() calls.
    """

    stepper = NodeStepper(self.sess, "optim")

    # First, call cont() on the two tensors on the intermediate level: e and f.
    result = stepper.cont("d:0")
    self.assertAllClose(2.0, result)
    self.assertEqual({}, stepper.last_feed_types())
    self.assertEqual(set(), stepper.dirty_variables())

    # The cont call above should have restored Variable "b".
    result = stepper.cont("e:0")
    self.assertAllClose(8.0, result)
    self.assertEqual({}, stepper.last_feed_types())
    self.assertEqual(set(), stepper.dirty_variables())

    # Now run update_a, so as to let Variable a be diry.
    result = stepper.cont("optim/update_a/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)
    self.assertEqual({"a:0"}, stepper.dirty_variables())

    # Now, run update_b.
    result = stepper.cont("optim/update_b/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)

    # The last cont() run should have use the handle of tensor e, but not the
    # handle of tensor d, because the transitive closure of e is clean, whereas
    # that of d is dirty due to the update to a in the previous cont() call.
    self.assertEqual({
        "e:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # The result of the update_b should be identical to as if no other
    # update_* cont() calls have occurred before.
    self.assertAllClose(1.0, self.sess.run(self.a))
    self.assertAllClose(1.84, self.sess.run(self.b))
    self.assertAllClose(4.0, self.sess.run(self.c))

  def testFinalize(self):
    """Test finalize() to restore variables and run the original fetch."""

    stepper = NodeStepper(self.sess, "optim")

    # Invoke update_b before calling finalize.
    stepper.cont("optim/update_b/ApplyGradientDescent",
                 restore_variable_values=True)

    result = stepper.finalize()
    self.assertIsNone(result)

    # The results of the Variable updates should be the same as if no cont()
    # call has occurred on update_b.
    self.assertAllClose(0.84, self.sess.run(self.a))
    self.assertAllClose(1.84, self.sess.run(self.b))
    self.assertAllClose(3.96, self.sess.run(self.c))

  def testOverrideThenContToUpdate(self):
    """Test cont() to update nodes after overriding tensor values."""

    stepper = NodeStepper(self.sess, "optim")

    result = stepper.cont("d:0")
    self.assertAllClose(2.0, result)
    self.assertEqual({}, stepper.last_feed_types())
    self.assertEqual(set(), stepper.dirty_variables())
    self.assertEqual(["d:0"], stepper.handle_names())

    # Override the value from 1.0 to 10.0.
    stepper.override_tensor("a/read:0", 10.0)

    self.assertEqual(["a/read:0"], stepper.override_names())

    result = stepper.cont("optim/update_c/ApplyGradientDescent",
                          restore_variable_values=True)
    self.assertIsNone(result)

    # The last cont() call should have not used the tensor handle to d:0,
    # because the transitive closure of d:0 contains an override tensor.
    self.assertEqual({
        "a/read:0": NodeStepper.FEED_TYPE_OVERRIDE
    }, stepper.last_feed_types())

    # The tensor handle to d:0 should have been removed due to the dirty
    # transitive closure.
    self.assertEqual([], stepper.handle_names())

    # For this backprop on c, the overriding value of a/read:0 should have been
    # used:
    #   4.0 - learning_rate * a * b * b
    #     = 4.0 - 0.01 * 10.0 * 2.0 * 2.0 = 3.6.
    self.assertAllClose(3.6, self.sess.run(self.c))

    # Now remove the overriding value of a/read:0.
    stepper.remove_override("a/read:0")
    self.assertEqual([], stepper.override_names())

    # Obtain the tensor handle to d:0 again.
    result = stepper.cont("d:0")
    self.assertAllClose(2.0, result)
    self.assertEqual(["d:0"], stepper.handle_names())

    # Then call update_c again, without restoring c.
    result = stepper.cont(
        "optim/update_c/ApplyGradientDescent", restore_variable_values=False)
    self.assertIsNone(result)

    # This time, the d:0 tensor handle should have been used, because its
    # transitive closure is clean.
    self.assertEqual({
        "d:0": NodeStepper.FEED_TYPE_HANDLE
    }, stepper.last_feed_types())

    # For this backprop on c, the overriding value of a/read:0 should have been
    # used:
    #   3.6 - learning_rate * a * b * b
    #     = 3.6 - 0.01 * 1.0 * 2.0 * 2.0 = 3.56.
    self.assertAllClose(3.56, self.sess.run(self.c))


if __name__ == "__main__":
  googletest.main()
