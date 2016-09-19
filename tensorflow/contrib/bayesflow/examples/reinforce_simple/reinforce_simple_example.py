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
"""Simple examples of the REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


distributions = tf.contrib.distributions
sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor


def split_apply_merge(inp, partitions, fns):
  """Split input according to partitions.  Pass results through fns and merge.

  Args:
    inp: the input vector
    partitions: tensor of same length as input vector, having values 0, 1
    fns: the two functions.

  Returns:
    the vector routed, where routed[i] = fns[partitions[i]](inp[i])
  """
  new_inputs = tf.dynamic_partition(inp, partitions, len(fns))
  new_outputs = [fns[i](x) for i, x in enumerate(new_inputs)]
  new_indices = tf.dynamic_partition(
      tf.range(0, inp.get_shape()[0]), partitions, len(fns))
  return tf.dynamic_stitch(new_indices, new_outputs)


def plus_1(inputs):
  return inputs + 1.0


def minus_1(inputs):
  return inputs - 1.0


def build_split_apply_merge_model():
  """Build the Split-Apply-Merge Model.

  Route each value of input [-1, -1, 1, 1] through one of the
  functions, plus_1, minus_1.  The decision for routing is made by
  4 Bernoulli R.V.s whose parameters are determined by a neural network
  applied to the input.  REINFORCE is used to update the NN parameters.

  Returns:
    The 3-tuple (route_selection, routing_loss, final_loss), where:

      - route_selection is an int 4-vector
      - routing_loss is a float 4-vector
      - final_loss is a float scalar.
  """
  inputs = tf.constant([[-1.0], [-1.0], [1.0], [1.0]])
  targets = tf.constant([[0.0], [0.0], [0.0], [0.0]])
  paths = [plus_1, minus_1]
  weights = tf.get_variable("w", [1, 2])
  bias = tf.get_variable("b", [1, 1])
  logits = tf.matmul(inputs, weights) + bias

  # REINFORCE forward step
  route_selection = st.StochasticTensor(
      distributions.Categorical, logits=logits)

  # Accessing route_selection as a Tensor below forces a sample of
  # the Categorical distribution based on its logits.
  # This is equivalent to calling route_selection.value().
  #
  # route_selection.value() returns an int32 4-vector with random
  # values in {0, 1}
  # COPY+ROUTE+PASTE
  outputs = split_apply_merge(inputs, route_selection, paths)

  # flatten routing_loss to a row vector (from a column vector)
  routing_loss = tf.reshape(tf.square(outputs - targets), shape=[-1])

  # Total loss: score function loss + routing loss.
  # The score function loss (through `route_selection.loss(routing_loss)`)
  # returns:
  #  [stop_gradient(routing_loss) *
  #   route_selection.log_pmf(stop_gradient(route_selection.value()))],
  # where log_pmf has gradients going all the way back to weights and bias.
  # In this case, the routing_loss depends on the variables only through
  # "route_selection", which has a stop_gradient on it.  So the
  # gradient of the loss really come through the score function
  surrogate_loss = sg.surrogate_loss([routing_loss])
  final_loss = tf.reduce_sum(surrogate_loss)

  return (route_selection, routing_loss, final_loss)


class REINFORCESimpleExample(tf.test.TestCase):

  def testSplitApplyMerge(self):
    # Repeatability.  SGD has a tendency to jump around, even here.
    tf.set_random_seed(1)

    with self.test_session() as sess:
      # Use sampling to train REINFORCE
      with st.value_type(st.SampleAndReshapeValue(n=1)):
        (route_selection,
         routing_loss,
         final_loss) = build_split_apply_merge_model()

      sgd = tf.train.GradientDescentOptimizer(1.0).minimize(final_loss)

      tf.initialize_all_variables().run()

      for i in range(10):
        # Run loss and inference step.  This toy problem converges VERY quickly.
        (routing_loss_v, final_loss_v, route_selection_v, _) = sess.run(
            [routing_loss, final_loss, tf.identity(route_selection), sgd])
        print(
            "Iteration %d, routing loss: %s, final_loss: %s, "
            "route selection: %s"
            % (i, routing_loss_v, final_loss_v, route_selection_v))

      self.assertAllEqual([0, 0, 1, 1], route_selection_v)
      self.assertAllClose([0.0, 0.0, 0.0, 0.0], routing_loss_v)
      self.assertAllClose(0.0, final_loss_v)


if __name__ == "__main__":
  tf.test.main()
