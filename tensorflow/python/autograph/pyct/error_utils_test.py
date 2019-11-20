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
# ==============================================================================
"""Tests for error_utils module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.platform import test


class ErrorMetadataBaseTest(test.TestCase):

  def test_create_exception_default_constructor(self):

    class CustomError(Exception):
      pass

    em = error_utils.ErrorMetadataBase(
        callsite_tb=(),
        cause_metadata=None,
        cause_message='test message',
        source_map={})
    exc = em.create_exception(CustomError())
    self.assertIsInstance(exc, CustomError)
    self.assertIn('test message', str(exc))

  def test_create_exception_custom_constructor(self):

    class CustomError(Exception):

      def __init__(self):
        super(CustomError, self).__init__('test_message')

    em = error_utils.ErrorMetadataBase(
        callsite_tb=(),
        cause_metadata=None,
        cause_message='test message',
        source_map={})
    exc = em.create_exception(CustomError())
    self.assertIsNone(exc)

  def test_get_message_when_frame_info_code_is_none(self):
    callsite_tb = [
        ('/path/one.py', 11, 'test_fn_1', None),
        ('/path/two.py', 171, 'test_fn_2', 'test code'),
    ]
    cause_message = 'Test message'
    em = error_utils.ErrorMetadataBase(
        callsite_tb=callsite_tb,
        cause_metadata=None,
        cause_message=cause_message,
        source_map={})
    self.assertRegex(
        em.get_message(),
        re.compile('test_fn_1.*test_fn_2.*Test message', re.DOTALL))


if __name__ == '__main__':
  test.main()
