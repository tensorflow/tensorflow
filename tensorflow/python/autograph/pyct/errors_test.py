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
"""Tests for errors module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import errors
from tensorflow.python.platform import test


class ErrorMetadataBaseTest(test.TestCase):

  def test_get_message_when_frame_info_code_is_none(self):
    callsite_tb = [
        ('/usr/local/python/foo.py',
         96,
         'fake_function_name',
         None),
        ('/usr/local/python/two.py',
         1874,
         'another_function_name',
         'raise ValueError(str(e))')]
    cause_message = 'ValueError: Just a test.'
    em = errors.ErrorMetadataBase(
        callsite_tb=callsite_tb,
        cause_metadata=None,
        cause_message=cause_message,
        source_map={})
    self.assertRegex(
        em.get_message(),
        r'fake_function(.|\n)*another_function(.|\n)*Just a test')


if __name__ == '__main__':
  test.main()
