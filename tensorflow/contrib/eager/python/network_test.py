# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import network
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import core


# pylint: disable=not-callable
class MyNetwork(network.Network):

  def __init__(self):
    super(MyNetwork, self).__init__(name="abcd")
    self.l1 = self.add_layer(core.Dense(1, use_bias=False))

  def call(self, x):
    return self.l1(x)


class NetworkTest(test.TestCase):

  def testTrainableAttribute(self):
    net = network.Network()
    self.assertTrue(net.trainable)
    with self.assertRaises(AttributeError):
      net.trainable = False
    self.assertTrue(net.trainable)

  def testNetworkCall(self):
    net = MyNetwork()
    net(constant_op.constant([[2.0]]))  # Force variables to be created.
    self.assertEqual(1, len(net.trainable_variables))
    net.trainable_variables[0].assign([[17.0]])
    # TODO(josh11b): Support passing Python values to networks.
    result = net(constant_op.constant([[2.0]]))
    self.assertEqual(34.0, result.numpy())

  def testNetworkAsAGraph(self):
    self.skipTest("TODO(ashankar,josh11b): FIX THIS")
    # Verify that we're using ResourceVariables

  def testNetworkVariablesDoNotInterfere(self):
    self.skipTest("TODO: FIX THIS")
    net1 = MyNetwork()
    net2 = MyNetwork()

    one = constant_op.constant([[1.]])

    print(type(net1(one)))
    net2(one)

    net1.trainable_weights[0].assign(constant_op.constant([[1.]]))
    net2.trainable_weights[0].assign(constant_op.constant([[2.]]))

    print("NET1")
    print(net1.name)
    print(net1.variables)
    print(net1(one))

    print("NET2")
    print(net2.name)
    print(net2.variables)
    print(net2(one))


class SequentialTest(test.TestCase):

  def testTwoLayers(self):
    # Create a sequential network with one layer.
    net = network.Sequential([core.Dense(1, use_bias=False)])

    # Set that layer's weights so it multiplies by 3
    l1 = net.get_layer(index=0)
    net(constant_op.constant([[2.0]]))  # Create l1's variables
    self.assertEqual(1, len(l1.trainable_variables))
    l1.trainable_variables[0].assign([[3.0]])
    self.assertEqual(21.0, net(constant_op.constant([[7.0]])).numpy())

    # Add a second layer to the network.
    l2 = core.Dense(1, use_bias=False)
    net.add_layer(l2)

    # Set the second layer's weights so it multiplies by 11
    net(constant_op.constant([[2.0]]))  # Create l2's variables
    self.assertEqual(1, len(l2.trainable_variables))
    l2.trainable_variables[0].assign([[11.0]])
    self.assertEqual(231.0, net(constant_op.constant([[7.0]])).numpy())


if __name__ == "__main__":
  test.main()
