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
"""Tests for custom user ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf


class InvalidOpTest(tf.test.TestCase):

  def testBasic(self):
    library_filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                    'invalid_op.so')
    with self.assertRaises(tf.errors.InvalidArgumentError):
      tf.load_op_library(library_filename)


if __name__ == '__main__':
  tf.test.main()
