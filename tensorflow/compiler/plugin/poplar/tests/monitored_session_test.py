# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import monitored_session as ms
from tensorflow.python.training import training_util


class IpuMonitoredSessionTest(test_util.TensorFlowTestCase):
  def testMonitoredSession(self):
    random_seed.set_random_seed(1)

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [2, 2], name="a")
      pb = array_ops.placeholder(np.float32, [2, 2], name="b")
      output = pa + pb

    with ms.MonitoredSession(session_creator=ms.ChiefSessionCreator()) as sess:

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      fd = {pa: [[0., 0.], [1., 1.]], pb: [[2., 1.], [4., 5.]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[2., 1.], [5., 6.]])

  def testTrainingLoop(self):
    random_seed.set_random_seed(1)

    # Model
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        x = array_ops.placeholder(np.float32, [4, 1, 4], name="a")
        l = array_ops.placeholder(np.float32, [4, 1, 1], name="b")

        y = layers.dense(x, 1, activation=nn.sigmoid)

        loss = losses.log_loss(l, y)
        train_op = gradient_descent.GradientDescentOptimizer(0.1) \
                                   .minimize(loss)

        init = variables.global_variables_initializer()

    # Test data
    image_data = [[[1, 1, 1, 1]], [[2, 2, 2, 2]], [[3, 3, 3, 3]],
                  [[4, 4, 4, 4]]]
    label_data = [[[1]], [[2]], [[3]], [[4]]]

    # Run training.
    with ms.MonitoredTrainingSession(
        is_chief=True,
        chief_only_hooks=None,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
      sess.run(init)
      previous_loss = float("inf")
      for i in range(5):
        measured_loss, _ = sess.run([loss, train_op],
                                    feed_dict={
                                        x: image_data,
                                        l: label_data
                                    })
        self.assertTrue(measured_loss < previous_loss)
        previous_loss = measured_loss

  def testMonitoredSessionStopAtStepHook(self):
    random_seed.set_random_seed(1)

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [2, 2], name="a")
      pb = array_ops.placeholder(np.float32, [2, 2], name="b")
      output = pa + pb

    with variable_scope.variable_scope('gs', use_resource=True):
      training_util.create_global_step()

    hook = basic_session_run_hooks.StopAtStepHook(num_steps=2)

    with ms.MonitoredSession(
        session_creator=ms.ChiefSessionCreator(), hooks=[hook]) as sess:

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      fd = {pa: [[0., 0.], [1., 1.]], pb: [[2., 1.], [4., 5.]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[2., 1.], [5., 6.]])


if __name__ == "__main__":
  googletest.main()
