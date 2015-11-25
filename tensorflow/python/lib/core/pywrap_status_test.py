# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for SWIG wrapped brain::Status."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import googletest


class StatusTest(googletest.TestCase):

  def testException(self):
    with self.assertRaises(pywrap_tensorflow.StatusNotOK) as context:
      pywrap_tensorflow.NotOkay()
    self.assertEqual(context.exception.code, error_codes_pb2.INVALID_ARGUMENT)
    self.assertEqual(context.exception.error_message, 'Testing 1 2 3')
    self.assertEqual(None, pywrap_tensorflow.Okay(),
                     'Status wrapper should not return anything upon OK.')


if __name__ == '__main__':
  googletest.main()
