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
"""Tests for tensorflow.tools.dist_test.scripts.kubectl_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.tools.dist_test.scripts import kubectl_util


kubectl_util.WAIT_PERIOD_SECONDS = 1


class KubectlUtilTest(googletest.TestCase):

  @test.mock.patch.object(subprocess, 'check_output')
  @test.mock.patch.object(subprocess, 'check_call')
  def testCreatePods(self, mock_check_call, mock_check_output):
    mock_check_output.return_value = 'nonempty'
    kubectl_util.CreatePods('test_pod', 'test.yaml')
    mock_check_call.assert_called_once_with(
        ['kubectl', 'create', '--filename=test.yaml'])
    mock_check_output.assert_called_once_with(
        ['kubectl', 'get', 'pods', '-o', 'name', '-l',
         'name-prefix in (test_pod)'], universal_newlines=True)

  @test.mock.patch.object(subprocess, 'check_output')
  @test.mock.patch.object(subprocess, 'call')
  def testDeletePods(self, mock_check_call, mock_check_output):
    mock_check_output.return_value = ''
    kubectl_util.DeletePods('test_pod', 'test.yaml')
    mock_check_call.assert_called_once_with(
        ['kubectl', 'delete', '--filename=test.yaml'])
    mock_check_output.assert_called_once_with(
        ['kubectl', 'get', 'pods', '-o', 'name', '-l',
         'name-prefix in (test_pod)'], universal_newlines=True)

  @test.mock.patch.object(subprocess, 'check_output')
  def testWaitForCompletion(self, mock_check_output):
    # Test success
    mock_check_output.return_value = '\'0,0,\''
    self.assertTrue(kubectl_util.WaitForCompletion('test_pod'))

    # Test failure
    mock_check_output.return_value = '\'0,1,\''
    self.assertFalse(kubectl_util.WaitForCompletion('test_pod'))

    # Test timeout
    with self.assertRaises(kubectl_util.TimeoutError):
      mock_check_output.return_value = '\'0,,\''
      kubectl_util.WaitForCompletion('test_pod', timeout=5)


if __name__ == '__main__':
  googletest.main()
