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
"""logit_fn tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.estimator.python.estimator import logit_fns
from tensorflow.python.client import session
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class LogitFnTest(test.TestCase):

  def test_simple_call_logit_fn(self):
    def dummy_logit_fn(features, mode):
      if mode == model_fn.ModeKeys.TRAIN:
        return features['f1']
      else:
        return features['f2']
    features = {
        'f1': constant_op.constant([[2., 3.]]),
        'f2': constant_op.constant([[4., 5.]])
    }
    logit_fn_result = logit_fns.call_logit_fn(
        dummy_logit_fn, features, model_fn.ModeKeys.EVAL, 'fake_params',
        'fake_config')
    with session.Session():
      self.assertAllClose([[4., 5.]], logit_fn_result.eval())

  def test_should_return_tensor(self):

    def invalid_logit_fn(features, params):
      return {
          'tensor1': features['f1'] * params['input_multiplier'],
          'tensor2': features['f2'] * params['input_multiplier']
      }
    features = {
        'f1': constant_op.constant([[2., 3.]]),
        'f2': constant_op.constant([[4., 5.]])
    }
    params = {'learning_rate': 0.001, 'input_multiplier': 2.0}
    with self.assertRaisesRegexp(ValueError, 'model_fn should return a Tensor'):
      logit_fns.call_logit_fn(invalid_logit_fn, features, 'fake_mode', params,
                              'fake_config')


if __name__ == '__main__':
  test.main()
