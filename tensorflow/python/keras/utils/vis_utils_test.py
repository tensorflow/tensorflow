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

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.keras.applications import efficientnet
from tensorflow.python.keras.utils import vis_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ModelToDotFormatTest(test.TestCase, parameterized.TestCase):

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

  @parameterized.named_parameters(
    (['block1a_project_conv', 'block1a_activation']),
    (['block1a_activation', 'block1a_project_conv']),
    (['block*', 'block2a_se_excite']),
    (['block*', 'block*']),
    (['block\da_activation', 'block\da_project_bn'])
  )
  def test_dot_layer_range(self, layer_range):
    model = efficientnet.EfficientNetB0()
    layer_ids_from_model = get_layer_ids_from_model(
      model, layer_range
    )
    try:
      dot = vis_utils.model_to_dot(model, layer_range=layer_range)
      dot_edges = dot.get_edges()
      layer_ids_from_dot = get_layer_ids_from_dot(
        dot_edges
      )
      self.assertAllEqual(sorted(layer_ids_from_model),
        sorted(layer_ids_from_dot))
    except ImportError:
      pass

  @parameterized.named_parameters(
    (['block1a_project_conv', 'block1a_activation']),
    (['block1a_activation', 'block1a_project_conv']),
    (['block*', 'block2a_se_excite']),
    (['block\da_activation', 'block\da_project_bn'])
  )
  def test_plot_layer_range(self, layer_range):
    model = efficientnet.EfficientNetB0()
    effnet_subplot = 'model_effnet.png'
    try:
      vis_utils.plot_model(model,
        to_file=effnet_subplot,
        layer_range=layer_range)
      self.assertTrue(file_io.file_exists_v2(effnet_subplot))
    except ImportError:
      pass
    finally:
      file_io.delete_file_v2(effnet_subplot)

  @parameterized.named_parameters(
    (['block1a_se_squeeze', 'block2a_project_conv']),
    (['block\da_se_reshape', 'block*'])
  )
  def test_layer_range_assertion_fail(self, layer_range):
    model = efficientnet.EfficientNetB0()
    try:
      with self.assertRaises(AssertionError):
        vis_utils.model_to_dot(model, layer_range=layer_range)
      with self.assertRaises(AssertionError):
        vis_utils.plot_model(model, layer_range=layer_range)
    except ImportError:
      pass

  @parameterized.named_parameters(
    (['block1a_activation']),
    ([]),
    (['input_1', 'block1a_activation', 'block1a_project_conv']),
    ([9, 'block1a_activation']),
    ([9, 29]),
    (['block8a_se_reshape', 'block*'])
  )
  def test_layer_range_value_fail(self, layer_range):
    model = efficientnet.EfficientNetB0()
    try:
      with self.assertRaises(ValueError):
        vis_utils.model_to_dot(model, layer_range=layer_range)
      with self.assertRaises(ValueError):
        vis_utils.plot_model(model, layer_range=layer_range)
    except ImportError:
      pass

def get_layer_ids_from_model(model, layer_range):
  layer_range = vis_utils.get_layer_index_bound_by_layer_name(model,
    layer_range)
  layer_ids_from_model = []
  for i, layer in enumerate(model.layers):
    if i >= layer_range[0] and i <= layer_range[1]:
      layer_ids_from_model.append(str(id(layer)))
  return layer_ids_from_model

def get_layer_ids_from_dot(dot_edges):
  layer_ids_from_dot = []
  for edge in dot_edges:
    for pt in edge.obj_dict["points"]:
      if pt not in layer_ids_from_dot:
        layer_ids_from_dot.append(pt)
  return layer_ids_from_dot


if __name__ == '__main__':
  test.main()
