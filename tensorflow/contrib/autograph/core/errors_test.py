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

from tensorflow.contrib.autograph.core import errors
from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors as tf_errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def zero_div():
  return array_ops.constant(10, dtype=dtypes.int32) // 0


def zero_div_caller():
  a = zero_div() + 2
  return a


class RuntimeErrorsTest(test.TestCase):

  def setUp(self):
    self._fake_origin = origin_info.OriginInfo('new file', 'new func', 96, 0,
                                               'print("hello world!")')

  def test_error_replacement(self):
    _, zero_div_lineno = tf_inspect.getsourcelines(zero_div)
    src_map = {
        errors.CodeLocation(
            file_path=__file__, line_number=zero_div_lineno + 1):
            self._fake_origin
    }
    with self.assertRaises(errors.TfRuntimeError) as cm:
      z = zero_div_caller()
      zero_div_caller.ag_source_map = src_map
      with errors.improved_errors(zero_div_caller):
        with self.test_session() as sess:
          sess.run(z)
    expected = cm.exception
    current_traceback = expected.custom_traceback
    for frame in current_traceback:
      self.assertNotEqual('zero_div', frame[2])
    self.assertTrue(
        any(self._fake_origin.as_frame() == frame
            for frame in current_traceback))

  def test_error_not_found(self):
    src_map = {
        errors.CodeLocation(file_path=__file__, line_number=-1):
            self._fake_origin
    }
    with self.assertRaises(errors.TfRuntimeError) as cm:
      z = zero_div_caller()
      zero_div_caller.ag_source_map = src_map
      with errors.improved_errors(zero_div_caller):
        with self.test_session() as sess:
          sess.run(z)
    expected = cm.exception
    current_traceback = expected.custom_traceback
    self.assertTrue(any('zero_div' in frame[2] for frame in current_traceback))
    for frame in current_traceback:
      self.assertNotEqual(frame, self._fake_origin.as_frame())

  def test_rewriting_error(self):
    _, zero_div_lineno = tf_inspect.getsourcelines(zero_div)
    src_map = {
        errors.CodeLocation(
            file_path=__file__, line_number=zero_div_lineno + 1):
            None
    }
    with self.assertRaisesRegexp(tf_errors.InvalidArgumentError,
                                 'Integer division by zero'):
      z = zero_div_caller()
      zero_div_caller.ag_source_map = src_map
      with errors.improved_errors(zero_div_caller):
        with self.test_session() as sess:
          sess.run(z)

  def test_no_ag_source_map(self):
    with self.assertRaisesRegexp(
        ValueError,
        'converted_function must be the result of an autograph.to_graph call'):
      with errors.improved_errors(None):
        pass

  def test_bad_ag_source_map(self):
    with self.assertRaisesRegexp(
        ValueError,
        'converted_function must be the result of an autograph.to_graph call'):
      src_map = None
      zero_div_caller.ag_source_map = src_map
      with errors.improved_errors(None):
        pass


if __name__ == '__main__':
  test.main()
