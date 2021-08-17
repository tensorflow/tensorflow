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
"""Tests for Keras Vis utils."""

from tensorflow.python import keras
from tensorflow.python.keras.utils import vis_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ModelToDotFormatTest(test.TestCase):

  def test_plot_model_cnn(self):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=2, kernel_size=(2, 3), input_shape=(3, 5, 5), name='conv'))
    model.add(keras.layers.Flatten(name='flat'))
    model.add(keras.layers.Dense(5, name='dense'))
    dot_img_file = 'model_1.png'
    try:
      vis_utils.plot_model(
          model, to_file=dot_img_file, show_shapes=True, show_dtype=True)
      self.assertTrue(file_io.file_exists_v2(dot_img_file))
      file_io.delete_file_v2(dot_img_file)
    except ImportError:
      pass

  def test_plot_model_with_wrapped_layers_and_models(self):
    inputs = keras.Input(shape=(None, 3))
    lstm = keras.layers.LSTM(6, return_sequences=True, name='lstm')
    x = lstm(inputs)
    # Add layer inside a Wrapper
    bilstm = keras.layers.Bidirectional(
        keras.layers.LSTM(16, return_sequences=True, name='bilstm'))
    x = bilstm(x)
    # Add model inside a Wrapper
    submodel = keras.Sequential(
        [keras.layers.Dense(32, name='dense', input_shape=(None, 32))]
    )
    wrapped_dense = keras.layers.TimeDistributed(submodel)
    x = wrapped_dense(x)
    # Add shared submodel
    outputs = submodel(x)
    model = keras.Model(inputs, outputs)
    dot_img_file = 'model_2.png'
    try:
      vis_utils.plot_model(
          model,
          to_file=dot_img_file,
          show_shapes=True,
          show_dtype=True,
          expand_nested=True)
      self.assertTrue(file_io.file_exists_v2(dot_img_file))
      file_io.delete_file_v2(dot_img_file)
    except ImportError:
      pass

  def test_plot_model_with_add_loss(self):
    inputs = keras.Input(shape=(None, 3))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.add_loss(math_ops.reduce_mean(outputs))
    dot_img_file = 'model_3.png'
    try:
      vis_utils.plot_model(
          model,
          to_file=dot_img_file,
          show_shapes=True,
          show_dtype=True,
          expand_nested=True)
      self.assertTrue(file_io.file_exists_v2(dot_img_file))
      file_io.delete_file_v2(dot_img_file)
    except ImportError:
      pass

    model = keras.Sequential([
        keras.Input(shape=(None, 3)), keras.layers.Dense(1)])
    model.add_loss(math_ops.reduce_mean(model.output))
    dot_img_file = 'model_4.png'
    try:
      vis_utils.plot_model(
          model,
          to_file=dot_img_file,
          show_shapes=True,
          show_dtype=True,
          expand_nested=True)
      self.assertTrue(file_io.file_exists_v2(dot_img_file))
      file_io.delete_file_v2(dot_img_file)
    except ImportError:
      pass


if __name__ == '__main__':
  test.main()
