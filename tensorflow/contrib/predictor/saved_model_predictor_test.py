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

"""Tests for predictor.saved_model_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.predictor import saved_model_predictor
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_def_utils


KEYS_AND_OPS = (('sum', lambda x, y: x + y),
                ('product', lambda x, y: x * y,),
                ('difference', lambda x, y: x - y))

MODEL_DIR_NAME = 'contrib/predictor/test_export_dir'


class SavedModelPredictorTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    # Load a saved model exported from the arithmetic `Estimator`.
    # See `testing_common.py`.
    cls._export_dir = test.test_src_dir_path(MODEL_DIR_NAME)

  def testDefault(self):
    """Test prediction with default signature."""
    np.random.seed(1111)
    x = np.random.rand()
    y = np.random.rand()
    predictor = saved_model_predictor.SavedModelPredictor(
        export_dir=self._export_dir)
    output = predictor({'x': x, 'y': y})['outputs']
    self.assertAlmostEqual(output, x + y, places=3)

  def testSpecifiedSignatureKey(self):
    """Test prediction with spedicified signature key."""
    np.random.seed(1234)
    for signature_def_key, op in KEYS_AND_OPS:
      x = np.random.rand()
      y = np.random.rand()
      expected_output = op(x, y)

      predictor = saved_model_predictor.SavedModelPredictor(
          export_dir=self._export_dir,
          signature_def_key=signature_def_key)

      output_tensor_name = predictor.fetch_tensors['outputs'].name
      self.assertRegexpMatches(
          output_tensor_name,
          signature_def_key,
          msg='Unexpected fetch tensor.')

      output = predictor({'x': x, 'y': y})['outputs']
      self.assertAlmostEqual(
          expected_output, output, places=3,
          msg='Failed for signature "{}." '
          'Got output {} for x = {} and y = {}'.format(
              signature_def_key, output, x, y))

  def testSpecifiedSignature(self):
    """Test prediction with spedicified signature definition."""
    np.random.seed(4444)
    for key, op in KEYS_AND_OPS:
      x = np.random.rand()
      y = np.random.rand()
      expected_output = op(x, y)

      inputs = {
          'x': meta_graph_pb2.TensorInfo(
              name='inputs/x:0',
              dtype=types_pb2.DT_FLOAT,
              tensor_shape=tensor_shape_pb2.TensorShapeProto()),
          'y': meta_graph_pb2.TensorInfo(
              name='inputs/y:0',
              dtype=types_pb2.DT_FLOAT,
              tensor_shape=tensor_shape_pb2.TensorShapeProto())}
      outputs = {
          key: meta_graph_pb2.TensorInfo(
              name='outputs/{}:0'.format(key),
              dtype=types_pb2.DT_FLOAT,
              tensor_shape=tensor_shape_pb2.TensorShapeProto())}
      signature_def = signature_def_utils.build_signature_def(
          inputs=inputs,
          outputs=outputs,
          method_name='tensorflow/serving/regress')
      predictor = saved_model_predictor.SavedModelPredictor(
          export_dir=self._export_dir,
          signature_def=signature_def)

      output_tensor_name = predictor.fetch_tensors[key].name
      self.assertRegexpMatches(
          output_tensor_name,
          key,
          msg='Unexpected fetch tensor.')

      output = predictor({'x': x, 'y': y})[key]
      self.assertAlmostEqual(
          expected_output, output, places=3,
          msg='Failed for signature "{}". '
          'Got output {} for x = {} and y = {}'.format(key, output, x, y))

  def testSpecifiedTensors(self):
    """Test prediction with spedicified `Tensor`s."""
    np.random.seed(987)
    for key, op in KEYS_AND_OPS:
      x = np.random.rand()
      y = np.random.rand()
      expected_output = op(x, y)
      input_names = {'x': 'inputs/x:0',
                     'y': 'inputs/y:0'}
      output_names = {key: 'outputs/{}:0'.format(key)}
      predictor = saved_model_predictor.SavedModelPredictor(
          export_dir=self._export_dir,
          input_names=input_names,
          output_names=output_names)

      output_tensor_name = predictor.fetch_tensors[key].name
      self.assertRegexpMatches(
          output_tensor_name,
          key,
          msg='Unexpected fetch tensor.')

      output = predictor({'x': x, 'y': y})[key]
      self.assertAlmostEqual(
          expected_output, output, places=3,
          msg='Failed for signature "{}". '
          'Got output {} for x = {} and y = {}'.format(key, output, x, y))

  def testBadTagsFail(self):
    """Test that predictor construction fails for bad tags."""
    bad_tags_regex = ('.* could not be found in SavedModel')
    with self.assertRaisesRegexp(RuntimeError, bad_tags_regex):
      _ = saved_model_predictor.SavedModelPredictor(
          export_dir=self._export_dir,
          tags=('zomg, bad, tags'))

  def testSpecifiedGraph(self):
    """Test that the predictor remembers a specified `Graph`."""
    g = ops.Graph()
    predictor = saved_model_predictor.SavedModelPredictor(
        export_dir=self._export_dir,
        graph=g)
    self.assertEqual(predictor.graph, g)


if __name__ == '__main__':
  test.main()
