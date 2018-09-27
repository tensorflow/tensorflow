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
"""Tests for model_coverage_lib.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.lite.testing.model_coverage import model_coverage_lib as model_coverage
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class EvaluateFrozenGraph(test.TestCase):

  def _saveFrozenGraph(self, sess):
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    return graph_def_file

  def testFloat(self):
    with session.Session().as_default() as sess:
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
    filename = self._saveFrozenGraph(sess)

    model_coverage.test_frozen_graph(filename, ['Placeholder'], ['add'])

  def testMultipleOutputs(self):
    with session.Session().as_default() as sess:
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16], dtype=dtypes.float32, name='inputB')

      weight = constant_op.constant(-1.0, shape=[16, 16])
      bias = constant_op.constant(-1.0, shape=[16])
      layer = math_ops.matmul(in_tensor_1, weight) + bias
      _ = math_ops.reduce_mean(math_ops.square(layer - in_tensor_2))
    filename = self._saveFrozenGraph(sess)

    model_coverage.test_frozen_graph(filename, ['inputA', 'inputB'],
                                     ['add', 'Mean'])


class EvaluateSavedModel(test.TestCase):

  def testFloat(self):
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with session.Session().as_default() as sess:
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      out_tensor = in_tensor_1 + in_tensor_2

      inputs = {'x': in_tensor_1, 'y': in_tensor_2}
      outputs = {'z': out_tensor}
      saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    model_coverage.test_saved_model(saved_model_dir)


class EvaluateKerasModel(test.TestCase):

  def _getSingleInputKerasModel(self):
    """Returns single input Sequential tf.keras model."""
    keras.backend.clear_session()

    xs = [-1, 0, 1, 2, 3, 4]
    ys = [-3, -1, 1, 3, 5, 7]

    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.train_on_batch(xs, ys)
    return model

  def _saveKerasModel(self, model):
    try:
      fd, keras_file = tempfile.mkstemp('.h5')
      keras.models.save_model(model, keras_file)
    finally:
      os.close(fd)
    return keras_file

  def testFloat(self):
    model = self._getSingleInputKerasModel()
    keras_file = self._saveKerasModel(model)

    model_coverage.test_keras_model(keras_file)

  def testPostTrainingQuantize(self):
    model = self._getSingleInputKerasModel()
    keras_file = self._saveKerasModel(model)

    model_coverage.test_keras_model(keras_file, post_training_quantize=True)

  def testConverterMode(self):
    model = self._getSingleInputKerasModel()
    keras_file = self._saveKerasModel(model)

    model_coverage.test_keras_model(
        keras_file, converter_mode=lite.ConverterMode.TOCO_FLEX)


if __name__ == '__main__':
  test.main()
