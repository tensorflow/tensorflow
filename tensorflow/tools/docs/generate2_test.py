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
"""Tests for tensorflow.tools.docs.generate2."""

import os
import pathlib
import shutil
import types
from unittest import mock

from packaging import version
import tensorflow as tf
import yaml

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import generate2


class AutoModule(types.ModuleType):

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError()
    mod = AutoModule(name)
    setattr(self, name, mod)
    return mod


# Make a mock tensorflow package that won't take too long to test.
fake_tf = AutoModule('FakeTensorFlow')
fake_tf.Module = tf.Module  # pylint: disable=invalid-name
fake_tf.feature_column.nummeric_column = tf.feature_column.numeric_column
fake_tf.keras.Model = tf.keras.Model
fake_tf.keras.preprocessing = tf.keras.preprocessing
fake_tf.keras.layers.Layer = tf.keras.layers.Layer
fake_tf.keras.optimizers.Optimizer = tf.keras.optimizers.Optimizer
fake_tf.nn.sigmoid_cross_entropy_with_logits = (
    tf.nn.sigmoid_cross_entropy_with_logits
)
fake_tf.raw_ops.Add = tf.raw_ops.Add
fake_tf.raw_ops.Print = tf.raw_ops.Print  # op with no XLA support
fake_tf.summary.audio = tf.summary.audio
fake_tf.summary.audio2 = tf.summary.audio
fake_tf.__version__ = tf.__version__


class Generate2Test(googletest.TestCase):

  @mock.patch.object(generate2, 'tf', fake_tf)
  def test_end_to_end(self):
    generate2.MIN_NUM_FILES_EXPECTED = 1
    output_dir = pathlib.Path(googletest.GetTempDir())/'output'
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    generate2.build_docs(
        output_dir=output_dir,
        code_url_prefix='',
        search_hints=True,
    )

    raw_ops_page = (output_dir/'tf/raw_ops.md').read_text()
    self.assertIn('/tf/raw_ops/Add.md', raw_ops_page)

    toc = yaml.safe_load((output_dir / 'tf/_toc.yaml').read_text())
    self.assertEqual({
        'title': 'Overview',
        'path': '/tf_overview'
    }, toc['toc'][0]['section'][0])
    redirects = yaml.safe_load((output_dir / 'tf/_redirects.yaml').read_text())
    self.assertIn({'from': '/tf_overview', 'to': '/tf'}, redirects['redirects'])

    if version.parse(fake_tf.__version__) >= version.parse('2.14'):
      self.assertIn(
          '<a id=Add href="/tf/raw_ops/Add.md">Add</a> | ✔️ | ✔️ |', raw_ops_page
      )
      self.assertIn(
          '<a id=Print href="/tf/raw_ops/Print.md">Print</a> | ✔️ | ❌ |',
          raw_ops_page,
      )

if __name__ == '__main__':
  googletest.main()
