# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for serialization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import losses
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.utils import losses_utils, generic_utils
from tensorflow.python.platform import test
from tensorflow.python.util import serialization


class SerializationTests(test.TestCase):

  def test_serialize_dense(self):
    dense = core.Dense(3)
    dense(constant_op.constant([[4.]]))
    round_trip = json.loads(json.dumps(
        dense, default=serialization.get_json_type))
    self.assertEqual(3, round_trip["config"]["units"])

  def test_serialize_shape(self):
    round_trip = json.loads(json.dumps(
        tensor_shape.TensorShape([None, 2, 3]),
        default=serialization.get_json_type))
    self.assertIs(round_trip[0], None)
    self.assertEqual(round_trip[1], 2)

  @test_util.run_in_graph_and_eager_modes
  def test_serialize_sequential(self):
    model = sequential.Sequential()
    model.add(core.Dense(4))
    model.add(core.Dense(5))
    model(constant_op.constant([[1.]]))
    sequential_round_trip = json.loads(
        json.dumps(model, default=serialization.get_json_type))
    self.assertEqual(
        5, sequential_round_trip["config"]["layers"][1]["config"]["units"])

  @test_util.run_in_graph_and_eager_modes
  def test_serialize_model(self):
    x = input_layer.Input(shape=[3])
    y = core.Dense(10)(x)
    model = training.Model(x, y)
    model(constant_op.constant([[1., 1., 1.]]))
    model_round_trip = json.loads(
        json.dumps(model, default=serialization.get_json_type))
    self.assertEqual(
        10, model_round_trip["config"]["layers"][1]["config"]["units"])

  @test_util.run_in_graph_and_eager_modes
  def test_serialize_custom_model_compile(self):
    with generic_utils.custom_object_scope():

      @generic_utils.register_keras_serializable(package="dummy-package")
      class DummySparseCategoricalCrossentropyLoss(losses.LossFunctionWrapper):
        # This loss is identical equal to tf.keras.losses.SparseCategoricalCrossentropy
        def __init__(
            self,
            from_logits=False,
            reduction=losses_utils.ReductionV2.AUTO,
            name="dummy_sparse_categorical_crossentropy_loss",
        ):
          super(DummySparseCategoricalCrossentropyLoss, self).__init__(
              losses.sparse_categorical_crossentropy,
              name=name,
              reduction=reduction,
              from_logits=from_logits,
          )

      x = input_layer.Input(shape=[3])
      y = core.Dense(10)(x)
      model = training.Model(x, y)
      model.compile(
          loss=DummySparseCategoricalCrossentropyLoss(from_logits=True))
      model_round_trip = json.loads(
          json.dumps(model.loss, default=serialization.get_json_type))

      # check if class name with package scope
      self.assertEqual("dummy-package>DummySparseCategoricalCrossentropyLoss",
                       model_round_trip["class_name"])

      # check if configure is correctly
      self.assertEqual(True, model_round_trip["config"]["from_logits"])


if __name__ == "__main__":
  test.main()
