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
"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.platform import test


class EstimatorConstructorTest(test.TestCase):

  def test_config_must_be_a_run_config(self):
    with self.assertRaisesRegexp(ValueError, 'an instance of RunConfig'):
      estimator.Estimator(model_fn=None, config='NotARunConfig')

  def test_model_fn_must_be_provided(self):
    with self.assertRaisesRegexp(ValueError, 'model_fn.* must be'):
      estimator.Estimator(model_fn=None)

  def test_property_accessors(self):

    def model_fn(features, labels, params):
      _, _, _ = features, labels, params

    class FakeConfig(run_config.RunConfig):  # pylint: disable=g-wrong-blank-lines
      pass

    params = {'hidden_layers': [3, 4]}
    est = estimator.Estimator(
        model_fn=model_fn, model_dir='bla', config=FakeConfig(), params=params)
    self.assertTrue(isinstance(est.config, FakeConfig))
    self.assertEqual(params, est.params)
    self.assertEqual('bla', est.model_dir)

  def test_default_config(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn)
    self.assertTrue(isinstance(est.config, run_config.RunConfig))

  def test_default_model_dir(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn)
    self.assertTrue(est.model_dir is not None)

  def test_model_fn_args_must_include_features(self):

    def model_fn(x, labels):
      _, _ = x, labels

    with self.assertRaisesRegexp(ValueError, 'features'):
      estimator.Estimator(model_fn=model_fn)

  def test_model_fn_args_must_include_labels(self):

    def model_fn(features, y):
      _, _ = features, y

    with self.assertRaisesRegexp(ValueError, 'labels'):
      estimator.Estimator(model_fn=model_fn)

  def test_if_params_provided_then_model_fn_should_accept_it(self):

    def model_fn(features, labels):
      _, _ = features, labels

    estimator.Estimator(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'params'):
      estimator.Estimator(model_fn=model_fn, params={'hidden_layers': 4})

  def test_not_known_model_fn_args_without_default(self):

    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    with self.assertRaisesRegexp(ValueError, 'something'):
      estimator.Estimator(model_fn=model_fn)


if __name__ == '__main__':
  test.main()
