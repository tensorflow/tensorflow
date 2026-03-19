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
"""Extended slice operations."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def basic_ext_slice(n):
  return n[:, :], n[0, :], n[:, 0]


def basic_expand_dims(n):
  return n[:, tf.newaxis] - n[tf.newaxis, :]


def slice_of_application(n, x):
  return n(x)[:, tf.newaxis] - n(x)[tf.newaxis, :]


class ReferenceTest(reference_test_base.TestCase):

  def test_basic_ext_slice(self):
    self.assertFunctionMatchesEager(basic_ext_slice, tf.eye(3))

  def test_basic_expand_dims(self):
    self.assertFunctionMatchesEager(basic_expand_dims, tf.eye(3))

  def test_slice_of_application(self):
    self.assertFunctionMatchesEager(slice_of_application, lambda x: x,
                                    tf.eye(3))


if __name__ == '__main__':
  tf.test.main()
