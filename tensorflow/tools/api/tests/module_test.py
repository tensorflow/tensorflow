# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Smoke tests for tensorflow module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pkgutil

import tensorflow as tf

from tensorflow.python.platform import test


class ModuleTest(test.TestCase):

  def testCanLoadWithPkgutil(self):
    out = pkgutil.find_loader('tensorflow')
    self.assertIsNotNone(out)

  def testDocString(self):
    self.assertIn('TensorFlow', tf.__doc__)
    self.assertNotIn('Wrapper', tf.__doc__)

  def testDict(self):
    # Check that a few modules are in __dict__.
    self.assertIn('nn', tf.__dict__)
    self.assertIn('keras', tf.__dict__)
    self.assertIn('image', tf.__dict__)

  def testName(self):
    self.assertEqual('tensorflow', tf.__name__)


if __name__ == '__main__':
  test.main()
