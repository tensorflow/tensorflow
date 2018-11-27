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
"""Tests for errors module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.core import errors
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors as tf_errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def zero_div():
  x = array_ops.constant(10, dtype=dtypes.int32)
  return x // 0


def zero_div_caller():
  return zero_div()


class RuntimeErrorsTest(test.TestCase):

  def fake_origin(self, function, line_offset):
    _, lineno = tf_inspect.getsourcelines(function)
    filename = tf_inspect.getsourcefile(function)
    lineno += line_offset
    loc = origin_info.LineLocation(filename, lineno)
    origin = origin_info.OriginInfo(loc, 'test_function_name', 'test_code',
                                    'test_comment')
    return loc, origin

  def test_improved_errors_basic(self):
    loc, origin = self.fake_origin(zero_div, 2)
    zero_div_caller.ag_source_map = {loc: origin}

    ops = zero_div_caller()
    with self.assertRaises(errors.TfRuntimeError) as cm:
      with errors.improved_errors(zero_div_caller):
        with self.cached_session() as sess:
          self.evaluate(ops)

    for frame in cm.exception.custom_traceback:
      _, _, function_name, _ = frame
      self.assertNotEqual('zero_div', function_name)
    self.assertIn(origin.as_frame(), set(cm.exception.custom_traceback))

  def test_improved_errors_no_matching_lineno(self):
    loc, origin = self.fake_origin(zero_div, -1)
    zero_div_caller.ag_source_map = {loc: origin}

    ops = zero_div_caller()
    with self.assertRaises(errors.TfRuntimeError) as cm:
      with errors.improved_errors(zero_div_caller):
        with self.cached_session() as sess:
          self.evaluate(ops)

    all_function_names = set()
    for frame in cm.exception.custom_traceback:
      _, _, function_name, _ = frame
      all_function_names.add(function_name)
      self.assertNotEqual('test_function_name', function_name)
    self.assertIn('zero_div', all_function_names)

  def test_improved_errors_failures(self):
    loc, _ = self.fake_origin(zero_div, 2)
    zero_div_caller.ag_source_map = {loc: 'bogus object'}

    ops = zero_div_caller()
    with self.assertRaises(tf_errors.InvalidArgumentError):
      with errors.improved_errors(zero_div_caller):
        with self.cached_session() as sess:
          self.evaluate(ops)

  def test_improved_errors_validation(self):
    with self.assertRaisesRegexp(
        ValueError,
        'converted_function must be the result of an autograph.to_graph call'):
      errors.improved_errors(zero_div).__enter__()
    with self.assertRaisesRegexp(
        ValueError,
        'converted_function must be the result of an autograph.to_graph call'):
      zero_div_caller.ag_source_map = 'not a dict'
      errors.improved_errors(zero_div_caller).__enter__()


if __name__ == '__main__':
  test.main()
