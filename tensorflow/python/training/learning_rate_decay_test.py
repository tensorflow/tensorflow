"""Functional test for learning rate decay."""
import tensorflow.python.platform

from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import learning_rate_decay


class LRDecayTest(test_util.TensorFlowTestCase):

  def testContinuous(self):
    with self.test_session():
      step = 5
      decayed_lr = learning_rate_decay.exponential_decay(0.05, step, 10, 0.96)
      expected = .05 * 0.96 ** (5.0 / 10.0)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testStaircase(self):
    with self.test_session():
      step = state_ops.variable_op([], types.int32)
      assign_100 = state_ops.assign(step, 100)
      assign_1 = state_ops.assign(step, 1)
      assign_2 = state_ops.assign(step, 2)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96 ** (100 / 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testVariables(self):
    with self.test_session():
      step = variables.Variable(1)
      assign_1 = step.assign(1)
      assign_2 = step.assign(2)
      assign_100 = step.assign(100)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      variables.initialize_all_variables().run()
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96 ** (100 / 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)


if __name__ == "__main__":
  googletest.main()
