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
"""Simple call to construct a namedtuple."""

import collections

import tensorflow.compat.v1 as tf

from tensorflow.python.autograph.tests import reference_test_base


def inline_namedtuple(x):
  nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
  n = nt(a=1, b=x)
  return n


def external_namedtuple(x, nt):
  return nt(a=1, b=x)


class NamedTupleSubclass(collections.namedtuple('TestNamedTuple', ('a',))):

  def foo(self):
    return self.a + 1


def namedtuple_subclass(x):
  nt = NamedTupleSubclass(x)
  return nt.foo()


class ReferenceTest(reference_test_base.TestCase):

  def test_inline(self):
    self.assertFunctionMatchesEager(inline_namedtuple, 1)
    self.assertFunctionMatchesEager(inline_namedtuple, tf.constant(1))

  def test_external(self):
    nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
    self.assertFunctionMatchesEager(external_namedtuple, 1, nt)
    self.assertFunctionMatchesEager(external_namedtuple, tf.constant(1), nt)

  def test_subclass(self):
    self.assertFunctionMatchesEager(namedtuple_subclass, 1)
    self.assertFunctionMatchesEager(namedtuple_subclass, tf.constant(1))


if __name__ == '__main__':
  tf.test.main()
