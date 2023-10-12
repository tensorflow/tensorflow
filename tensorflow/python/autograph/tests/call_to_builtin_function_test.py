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
"""Simple call to a builtin function."""

import unittest

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


# TODO(mdan): Add tests for all builtins.


def dict_call(x):
  return dict(foo=x)


def dict_call_aliased(x):
  def fake_dict(x):
    return x

  dict = fake_dict  # pylint:disable=redefined-builtin
  return dict(x)


def dict_call_dynamic(x):
  def gen_dict():
    return dict

  d = gen_dict()
  return d(foo=x)


def len_call(x):
  return len(x)


def nested_call(x):
  return list(range(len(x)))


def nested_cast(x):
  return float(int(x))


def len_call_aliased(x):

  def fake_len(x):
    return x

  len = fake_len  # pylint:disable=redefined-builtin
  return len(x)


def len_call_dynamic(x):

  def gen_len():
    return len

  l = gen_len()
  return l(x)


def len_call_on_mock():
  x = unittest.mock.MagicMock()
  return len(x)


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertFunctionMatchesEager(dict_call, 1)
    self.assertFunctionMatchesEager(len_call, [1, 2])
    self.assertFunctionMatchesEager(dict_call_aliased, 1)
    self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
    self.assertFunctionMatchesEager(dict_call_dynamic, 1)
    self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
    self.assertFunctionMatchesEager(nested_call, [])
    self.assertFunctionMatchesEager(nested_call, [1, 2, 3])

  def test_basic_tensor(self):
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(dict_call, 1)
    self.assertFunctionMatchesEager(len_call, [1, 2])
    self.assertFunctionMatchesEager(dict_call_aliased, 1)
    self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
    self.assertFunctionMatchesEager(dict_call_dynamic, 1)
    self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
    self.assertFunctionMatchesEager(nested_call, [])
    self.assertFunctionMatchesEager(nested_call, [1, 2, 3])


if __name__ == '__main__':
  tf.test.main()
