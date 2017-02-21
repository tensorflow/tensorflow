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

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.training.python.training import sampling_ops
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class SamplingOpsThreadingTest(test.TestCase):

  def testMultiThreadedEstimateDataDistribution(self):
    num_classes = 10

    # Set up graph.
    random_seed.set_random_seed(1234)
    label = math_ops.cast(
        math_ops.round(random_ops.random_uniform([1]) * num_classes),
        dtypes_lib.int32)

    prob_estimate = sampling_ops._estimate_data_distribution(  # pylint: disable=protected-access
        label, num_classes)
    # Check that prob_estimate is well-behaved in a multithreaded context.
    _, _, [prob_estimate] = sampling_ops._verify_input(  # pylint: disable=protected-access
        [], label, [prob_estimate])

    # Use queues to run multiple threads over the graph, each of which
    # fetches `prob_estimate`.
    queue = data_flow_ops.FIFOQueue(
        capacity=25,
        dtypes=[prob_estimate.dtype],
        shapes=[prob_estimate.get_shape()])
    enqueue_op = queue.enqueue([prob_estimate])
    queue_runner_impl.add_queue_runner(
        queue_runner_impl.QueueRunner(queue, [enqueue_op] * 25))
    out_tensor = queue.dequeue()

    # Run the multi-threaded session.
    with self.test_session() as sess:
      # Need to initialize variables that keep running total of classes seen.
      variables.global_variables_initializer().run()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      for _ in range(25):
        sess.run([out_tensor])

      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  test.main()
