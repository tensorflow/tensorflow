# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pickling / deepcopying of Keras Models."""

import copy
import pickle

import numpy as np

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers as layers_module
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.platform import test


class TestPickleProtocol(keras_parameterized.TestCase):
  """Tests pickle protoocol support.
  """

  @keras_parameterized.run_all_keras_modes
  def test_pickle_model(self):
    """Test copy.copy, copy.deepcopy and pickle on Functional Model."""

    def roundtrip(model):
      model = copy.copy(model)
      model = copy.deepcopy(model)
      for protocol in range(5):  # support up to protocol version 5
        model = pickle.loads(pickle.dumps(model, protocol=protocol))
      return model

    # create model
    x = input_layer.Input((3,))
    y = layers_module.Dense(2)(x)
    original_model = training_module.Model(x, y)
    model = original_model
    original_weights = model.get_weights()
    # roundtrip without compiling
    model = roundtrip(model)
    # compile
    model.compile(optimizer='sgd', loss='mse')
    # roundtrip compiled but not trained
    model = roundtrip(model)
    # train
    x = np.random.random((1000, 3))
    y = np.random.random((1000, 2))
    model.fit(x, y)
    y1 = model.predict(x)
    # roundtrip with training
    model = roundtrip(model)
    y2 = model.predict(x)
    self.assertAllClose(y1, y2)
    # check that the original model has not been changed
    final_weights = original_model.get_weights()
    self.assertAllClose(original_weights, final_weights)
    self.assertNotAllClose(original_weights, model.get_weights())

if __name__ == '__main__':
  test.main()
