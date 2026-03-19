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
"""Generators."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def basic_generator():
  yield 1


def generator_in_for(n):
  for i in range(n):
    yield i


def generator_in_while(n):
  i = 0
  while i < n:
    i += 1
    yield i


class LoopControlFlowTest(reference_test_base.TestCase):

  def test_basic_generator(self):
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(basic_generator)()

  def test_generator_in_for(self):
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(generator_in_for)([])

  def test_generator_in_while(self):
    with self.assertRaisesRegex(NotImplementedError, 'generators'):
      tf.function(generator_in_while)(0)


if __name__ == '__main__':
  tf.test.main()

