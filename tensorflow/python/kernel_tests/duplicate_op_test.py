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

import os

from tensorflow.python.framework import load_library
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class DuplicateOpTest(test.TestCase):

  def testBasic(self):
    library_filename = os.path.join(resource_loader.get_data_files_path(),
                                    'duplicate_op.so')
    duplicate = load_library.load_op_library(library_filename)

    self.assertEqual(len(duplicate.OP_LIST.op), 0)

    with self.cached_session():
      self.assertEqual(math_ops.add(1, 41).eval(), 42)


if __name__ == '__main__':
  test.main()
