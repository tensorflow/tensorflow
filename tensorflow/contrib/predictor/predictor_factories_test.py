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
"""Tests for predictor.predictor_factories."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.predictor import predictor_factories
from tensorflow.contrib.predictor import testing_common
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import test

MODEL_DIR_NAME = 'contrib/predictor/test_export_dir'


class PredictorFactoriesTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    # Load a saved model exported from the arithmetic `Estimator`.
    # See `testing_common.py`.
    cls._export_dir = test.test_src_dir_path(MODEL_DIR_NAME)

  def testFromSavedModel(self):
    """Test loading from_saved_model."""
    predictor_factories.from_saved_model(self._export_dir)

  def testFromSavedModelWithTags(self):
    """Test loading from_saved_model with tags."""
    predictor_factories.from_saved_model(self._export_dir, tags='serve')

  def testFromSavedModelWithSessionConfig(self):
    """Test loading from_saved_model with session config."""
    predictor_factories.from_saved_model(
        self._export_dir, config=config_pb2.ConfigProto())

  def testFromSavedModelWithBadTags(self):
    """Test that loading fails for bad tags."""
    bad_tags_regex = ('.*? could not be found in SavedModel')
    with self.assertRaisesRegexp(RuntimeError, bad_tags_regex):
      predictor_factories.from_saved_model(self._export_dir, tags='bad_tag')

  def testFromContribEstimator(self):
    estimator = testing_common.get_arithmetic_estimator(core=False)
    input_fn = testing_common.get_arithmetic_input_fn(core=False)
    predictor_factories.from_contrib_estimator(
        estimator, input_fn, output_alternative_key='sum')

  def testFromContribEstimatorWithSessionConfig(self):
    estimator = testing_common.get_arithmetic_estimator(core=False)
    input_fn = testing_common.get_arithmetic_input_fn(core=False)
    predictor_factories.from_contrib_estimator(
        estimator, input_fn, output_alternative_key='sum',
        config=config_pb2.ConfigProto())

  def testFromContribEstimatorWithCoreEstimatorRaises(self):
    estimator = testing_common.get_arithmetic_estimator(core=True)
    input_fn = testing_common.get_arithmetic_input_fn(core=True)
    with self.assertRaises(TypeError):
      predictor_factories.from_contrib_estimator(estimator, input_fn)

  def testFromCoreEstimator(self):
    estimator = testing_common.get_arithmetic_estimator(core=True)
    input_fn = testing_common.get_arithmetic_input_fn(core=True)
    predictor_factories.from_estimator(estimator, input_fn)

  def testFromCoreEstimatorWithSessionConfig(self):
    estimator = testing_common.get_arithmetic_estimator(core=True)
    input_fn = testing_common.get_arithmetic_input_fn(core=True)
    predictor_factories.from_estimator(
        estimator, input_fn, config=config_pb2.ConfigProto())

  def testFromCoreEstimatorWithContribEstimatorRaises(self):
    estimator = testing_common.get_arithmetic_estimator(core=False)
    input_fn = testing_common.get_arithmetic_input_fn(core=False)
    with self.assertRaises(TypeError):
      predictor_factories.from_estimator(estimator, input_fn)


if __name__ == '__main__':
  test.main()
