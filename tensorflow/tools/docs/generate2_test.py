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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import types
from unittest import mock

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import generate2

# Make a mock tensorflow package that won't take too long to test.
fake_tf = types.ModuleType('FakeTensorFlow')
fake_tf.estimator = tf.estimator
fake_tf.keras = tf.keras
fake_tf.nn = tf.nn
fake_tf.summary = tf.summary
fake_tf.raw_ops = types.ModuleType('raw_ops')
fake_tf.Module = tf.Module

for name in sorted(dir(tf.raw_ops))[:5]:
  setattr(fake_tf.raw_ops, name, getattr(tf.raw_ops, name))


class Generate2Test(googletest.TestCase):

  @mock.patch.object(generate2, 'tf', fake_tf)
  def test_end_to_end(self):
    output_dir = os.path.join(googletest.GetTempDir(), 'output')
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with self.assertRaisesRegex(ValueError, '2000 files'):
      generate2.build_docs(
          output_dir=output_dir,
          code_url_prefix='',
          search_hints=True,
          gen_report=False,
      )


if __name__ == '__main__':
  googletest.main()
