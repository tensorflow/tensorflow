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
"""Tests for ExportStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.python.platform import test


class ExportStrategyTest(test.TestCase):

  def test_no_optional_args_export(self):
    model_path = '/path/to/model'
    def _export_fn(estimator, export_path):
      self.assertTupleEqual((estimator, export_path), (None, None))
      return model_path

    strategy = export_strategy.ExportStrategy('foo', _export_fn)
    self.assertTupleEqual(strategy, ('foo', _export_fn, None))
    self.assertIs(strategy.export(None, None), model_path)

  def test_checkpoint_export(self):
    ckpt_model_path = '/path/to/checkpoint_model'
    def _ckpt_export_fn(estimator, export_path, checkpoint_path):
      self.assertTupleEqual((estimator, export_path), (None, None))
      self.assertEqual(checkpoint_path, 'checkpoint')
      return ckpt_model_path

    strategy = export_strategy.ExportStrategy('foo', _ckpt_export_fn)
    self.assertTupleEqual(strategy, ('foo', _ckpt_export_fn, None))
    self.assertIs(strategy.export(None, None, 'checkpoint'), ckpt_model_path)

  def test_checkpoint_eval_export(self):
    ckpt_eval_model_path = '/path/to/checkpoint_eval_model'
    def _ckpt_eval_export_fn(estimator, export_path, checkpoint_path,
                             eval_result):
      self.assertTupleEqual((estimator, export_path), (None, None))
      self.assertEqual(checkpoint_path, 'checkpoint')
      self.assertEqual(eval_result, 'eval')
      return ckpt_eval_model_path

    strategy = export_strategy.ExportStrategy('foo', _ckpt_eval_export_fn)
    self.assertTupleEqual(strategy, ('foo', _ckpt_eval_export_fn, None))
    self.assertIs(strategy.export(None, None, 'checkpoint', 'eval'),
                  ckpt_eval_model_path)

  def test_eval_only_export(self):
    def _eval_export_fn(estimator, export_path, eval_result):
      del estimator, export_path, eval_result

    strategy = export_strategy.ExportStrategy('foo', _eval_export_fn)
    self.assertTupleEqual(strategy, ('foo', _eval_export_fn, None))
    with self.assertRaisesRegexp(ValueError, 'An export_fn accepting '
                                 'eval_result must also accept '
                                 'checkpoint_path'):
      strategy.export(None, None, eval_result='eval')

  def test_strip_default_attr_export(self):
    strip_default_attrs_model_path = '/path/to/strip_default_attrs_model'
    def _strip_default_attrs_export_fn(estimator, export_path,
                                       strip_default_attrs):
      self.assertTupleEqual((estimator, export_path), (None, None))
      self.assertTrue(strip_default_attrs)
      return strip_default_attrs_model_path

    strategy = export_strategy.ExportStrategy('foo',
                                              _strip_default_attrs_export_fn,
                                              True)
    self.assertTupleEqual(strategy,
                          ('foo', _strip_default_attrs_export_fn, True))
    self.assertIs(strategy.export(None, None), strip_default_attrs_model_path)

if __name__ == '__main__':
  test.main()
