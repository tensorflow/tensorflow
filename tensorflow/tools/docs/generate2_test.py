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

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import generate2

# Make a mock tensorflow package that won't take too long to test.
fake_tf = types.ModuleType('FakeTensorFlow')
fake_tf.estimator = tf_estimator
fake_tf.feature_column = tf.feature_column
fake_tf.keras = tf.keras
fake_tf.nn = tf.nn
fake_tf.summary = tf.summary
fake_tf.raw_ops = types.ModuleType('raw_ops')
fake_tf.raw_ops.Add = tf.raw_ops.Add
fake_tf.Module = tf.Module
fake_tf.__version__ = tf.__version__

for name in sorted(dir(tf.raw_ops))[:5]:
  setattr(fake_tf.raw_ops, name, getattr(tf.raw_ops, name))


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


if __name__ == '__main__':
  googletest.main()
